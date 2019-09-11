"""
Splits an input protein to 32x32x32 patches, runs the model on them and returns a	helix prediction for the whole protein.
"""
import dataset_manager
import model
import numpy
import torch


def run_net_on_whole_protein(net, protein_data):
    """
    Gets a neural network and a 3d-list representing densities of a cryo-EM map.
    The function creates an output list of the same dimensions of the map, goes through almost disjoint 32x32x32 cubes of
    the given map with steps of 29 voxels, and actiates the prediction for the cube. The function saves the results of each
    voxel in its corresponding voxel in the output list. The value that is stored to a voxel is the value that was predicted
    to the voxel when it was as close to the center of cube as possible.
    :param net: A trained neural network
    :param protein_data: A 3d list containing the densities of voxels of a 3d cryo-EM map
    :return: a 3d-list with same dimensions as the given cryo-EM map, that contains the prediction of the net for each voxel
    """
    size = dataset_manager.dimensions(protein_data)
    label_data = [[[0 for j in range(size[2])] for i in range(size[1])] for k in range(size[0])]
    distance_from_cube_center = [[[10000000 for j in range(size[2])] for i in range(size[1])] for k in range(size[0])]
    for i_start in range(0, size[0]-16, 29):
        for j_start in range(0, size[1]-16, 29):
            for k_start in range(0, size[2]-16, 29):
                patch = dataset_manager.get_cube(protein_data, 32, i_start, j_start, k_start)
                labels = model.run_net_on_patch(net, patch)
                for i_cur in range(i_start, i_start + 32):
                    for j_cur in range(j_start, j_start + 32):
                        for k_cur in range(k_start, k_start + 32):
                            if i_cur < size[0] and j_cur < size[1] and k_cur < size[2]:
                                new_distance = abs(15.5-i_cur+i_start) + abs(15.5-j_cur+j_start) + abs(15.5-k_cur+k_start)
                                if new_distance < distance_from_cube_center[i_cur][j_cur][k_cur]:
                                    label_data[i_cur][j_cur][k_cur] = labels[i_cur - i_start][j_cur - j_start][k_cur - k_start]
                                    distance_from_cube_center[i_cur][j_cur][k_cur] = new_distance

    return numpy.asarray(label_data).astype('float32')


"""
TO DELETE
"""
def calc_stats(output, label):
    fp=0
    fn=0
    tp=0
    tn=0
    size = dataset_manager.dimensions(label)
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            for k in range(0, size[2]):
                if output[i][j][k] > 0.5 and label[i][j][k] == 1:
                    tp += 1
                elif output[i][j][k] <= 0.5 and label[i][j][k] == 1:
                    fn += 1
                elif output[i][j][k] > 0.5 and label[i][j][k] == 0:
                    fp += 1
                elif output[i][j][k] <= 0.5 and label[i][j][k] == 0:
                    tn += 1
    print(f'tn: {tn}, tp: {tp}, fn: {fn}, fp: {fp}, recall: {tp / (tp + fn)}, precision: {tp / (tp + fp)}')


def run_net_on_mrc_and_save(net_path, protein_mrc_path, output_mrc_path):
    """
    loads a trained neural network, reads a cryo-EM map, runs the neural network on the cryo-EM map, applies a cutoff
    to the resulted map in order to round the values to either 0 or 1, and saves the resulted map to an .mrc file
    :param net_path: path of the trained neural network
    :param protein_mrc_path: path of the input cryo-EM map
    :param output_mrc_path: path to save the .mrc prediction
    """
    net = model.load_net(net_path)
    protein_data = torch.Tensor(dataset_manager.read_mrc(protein_mrc_path)).to('cuda')
    label_data = run_net_on_whole_protein(net, protein_data)
    true_label_data = torch.Tensor(dataset_manager.read_mrc(protein_mrc_path.split('.')[0] + '_helix.mrc')).to('cuda') # to delete later
    dataset_manager.apply_cutoff(true_label_data,0.25)
    calc_stats(label_data, true_label_data) # to delete later
    dataset_manager.save_mrc(label_data.detach().numpy(), output_mrc_path)
