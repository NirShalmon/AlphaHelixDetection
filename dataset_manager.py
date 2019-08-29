import mrcfile, os, numpy
import random


def read_mrc(path):
    file = mrcfile.open(path)
    data = file.data.tolist()
    file.close()
    return data


def save_mrc(data, path):
    file = mrcfile.new(path, overwrite=True)
    # print(data[0][0][0])
    file.set_data(data)
    file.close()


def dimentions(protein_map):
    '''
	Returns a tuple of the size of a map
	'''
    return (len(protein_map), len(protein_map[0]), len(protein_map[0][0]))


def check_same_size(a, b):
    return dimentions(a) == dimentions(b)


def apply_cutoff(protein_map, cutoff):
    size = dimentions(protein_map)
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                if protein_map[i][j][k] >= cutoff:
                    protein_map[i][j][k] = 1
                else:
                    protein_map[i][j][k] = 0


def get_cube(protein_map, size, i, j, k):
    '''
    returns a size*size*size list of values from map, starting from map[i][j][k]
    pads with zeroes if neccesary
    '''
    cube = [[[0 for j in range(size)] for i in range(size)] for k in range(size)]
    map_size = dimentions(protein_map)
    for ii in range(i, size + i):
        for jj in range(j, size + j):
            for kk in range(k, size + k):
                if ii < map_size[0] and jj < map_size[1] and kk < map_size[2]:
                    cube[ii - i][jj - j][kk - k] = protein_map[ii][jj][kk]
    return cube


def get_dataset(protein_path, max_protein_amount=-1):
    protein_number = 0
    for filename in os.listdir(protein_path):
        if not filename.endswith('_helix.mrc'):
            continue
        protein_number += 1
        if max_protein_amount != -1 and protein_number > max_protein_amount:
            break
        protein_name = filename[:filename.index('_')]
        protein_map = read_mrc(os.path.join(protein_path, protein_name + '.mrc'))
        helix_map = read_mrc(os.path.join(protein_path, filename))
        apply_cutoff(helix_map, 0.25)
        sz = dimentions(protein_map)
        print(sz)
        for i in range(0, max(1, sz[0] - 32), 16):
            for j in range(0, max(1, sz[1] - 32), 16):
                for k in range(0, max(1, sz[2] - 32), 16):
                    yield (get_cube(protein_map, 32, i, j, k), get_cube(helix_map, 32, i, j, k))


def split_training_validation_sets(dataset):
    random.shuffle(dataset)
    return dataset[:len(dataset) * 7 // 10], dataset[len(dataset) * 7 // 10:]
