"""
Converts Cryo-EM maps (.mrc files) to datasets: training sets and validation sets.
"""
import random

import mrcfile
import os
import torch


def read_mrc(path):
    """
    Reads an mrc file into a 3D list of it's data
    :param path: The path to the mrc file
    :return: A 3D list of the data in the mrc file
    """
    file = mrcfile.open(path)
    data = file.data.tolist()
    file.close()
    return data


def save_mrc(data, path):
    """
    Saves an mrc file.
    :param data: The data to save in th mrc file, as a numpy array
    :param path: The path where the file would be saved. This will overwrite 
    """
    file = mrcfile.new(path, overwrite=True)
    file.set_data(data)
    file.close()


def dimensions(protein_map):
    """
    Returns a tuple representing the size of the input map's data
    :param protein_map: A 3D list of protein density data.
    :return: A tuple (size in dimension 0, size in dimension 1, size in dimension 2)
    """
    return len(protein_map), len(protein_map[0]), len(protein_map[0][0])


def apply_cutoff(protein_map, cutoff):
    """
    Converts a density map to a binary mask, by setting all cells of value >= cutoff to 1 and the rest to 0
    Assumes 0<cutoff<1
    :param protein_map: The density map to apply the cutoff on
    :param cutoff: The lowest density value that will be converted to 1 in the map.
    :returns: The protein mask after the cutoff
    """
    clamped_map = torch.clamp(protein_map, 0, 1)
    return torch.ceil(clamped_map-cutoff)


def get_cube(protein_map, size, i_start, j_start, k_start):
    """
    Returns a size*size*size list of values from map, from protein_map[i_start][j_start][k_start]
    to protein_map[i_start+size-1][j_start+size-1][k_start+size-1].
    Pads with zeroes if necessary
    :param protein_map: The protein's data as a 3D list.
    """
    cube = torch.empty(32, 32, 32)
    map_size = dimensions(protein_map)
    # The size of the part of the protein_map that fits in the patch without padding
    unpadded_size = (min(32, map_size[0]-i_start), min(32, map_size[1]-j_start), min(32, map_size[2]-k_start))
    cube[:unpadded_size[0], :unpadded_size[1], :unpadded_size[2]] = protein_map[i_start:i_start+unpadded_size[0], j_start:j_start+unpadded_size[1], k_start:k_start+unpadded_size[2]]
    return cube


def get_dataset(protein_path, max_protein_amount=-1):
    """
    Reads protein electron density data from mrc files and corresponding helix data from mrc files and returns a dataset
    of tuples of 32x32x32 patches of protein density data and corresponding 32x32x32 patches of helix data.
    The exact format is explained in the user manual.
    :param protein_path: The path where the formula
    :param max_protein_amount: No more than max_protein_amount proteins will be used in the dataset, unless max_protein_amount==-1.
    """
    print('Reading dataset')
    protein_number = 0  # The index of the current protein in the for loop.
    for filename in os.listdir(protein_path):
        if not filename.endswith('_helix.mrc') or filename == '1qvu_helix.mrc':  # Start by reading helix data.
            continue
        protein_number += 1
        if max_protein_amount != -1 and protein_number > max_protein_amount:
            break
        protein_name = filename[:filename.index('_')]  # The file name is protein-name_helix.mrc
        protein_map = torch.Tensor(read_mrc(os.path.join(protein_path, protein_name + '.mrc')))  # Read the protein density map
        helix_map = torch.Tensor(read_mrc(os.path.join(protein_path, filename)))
        apply_cutoff(helix_map, 0.25)
        sz = dimensions(protein_map)
        # Get patches of size 32x32x32 from the protein, with some overlap
        for i in range(0, max(1, sz[0] - 32), 16):
            for j in range(0, max(1, sz[1] - 32), 16):
                for k in range(0, max(1, sz[2] - 32), 16):
                    yield (get_cube(protein_map, 32, i, j, k), get_cube(helix_map, 32, i, j, k))


def split_training_validation_sets(dataset):
    """
    Splits the dataset created using get_dataset to a training set and a validation sed.
    :param dataset: The dataset created with get_dataset
    :return: The tuple (training_set, validation_set). Each dataset is of the same format of the datasets from get_dataset.
    The training set will contain 70% of the original dataset, and the validation set will contains the remaining 30%.
    """
    random.shuffle(dataset)
    return dataset[:len(dataset) * 7 // 10], dataset[len(dataset) * 7 // 10:]
