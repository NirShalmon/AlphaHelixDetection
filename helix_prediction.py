"""
Splits an input protein to 32x32x32 patches, runs the model on them and returns a	helix prediction for the whole protein.
Editor: Dana Milan
"""
import dataset_manager
import model
import numpy
import torch

CUBE_SIZE = dataset_manager.CUBE_SIZE

def run_net_on_whole_protein(net, protein_data):
    """
    Gets a neural network and a 3d-list representing densities of a cryo-EM map.
    The function creates an output list of the same dimensions of the map, goes through almost disjoint 32x32x32 cubes of
    the given map with steps of 29 voxels, and actiates the prediction for the cube. The function saves the results of each
    voxel in its corresponding voxel in the output list.
    :param net: A trained neural network
    :param protein_data: A 3d list containing the densities of voxels of a 3d cryo-EM map
    :return: a 3d-list with same dimensions as the given cryo-EM map, that contains the prediction of the net for each voxel
    """
    size = dataset_manager.dimensions(protein_data)
    computed_helices = numpy.empty([size[0],size[1],size[2]])
    for i_start in range(0, size[0]-CUBE_SIZE//2, CUBE_SIZE):
        for j_start in range(0, size[1]-CUBE_SIZE//2, CUBE_SIZE):
            for k_start in range(0, size[2]-CUBE_SIZE//2, CUBE_SIZE):
                patch = dataset_manager.get_cube(protein_data, i_start, j_start, k_start)
                patch_helices = model.run_net_on_patch(net, patch)
                computed_helices[i_start:min(i_start+CUBE_SIZE, size[0]),j_start:min(j_start+CUBE_SIZE, size[1]),k_start:min(k_start+CUBE_SIZE, size[2])] = \
                patch_helices[:min(size[0]-i_start, CUBE_SIZE),:min(size[1]-j_start,CUBE_SIZE),:min(size[2]-k_start,CUBE_SIZE)].cpu().detach().numpy()
    return computed_helices.astype('float32')


def run_net_on_mrc_and_save(net_path, protein_mrc_path, prob_output_mrc_path = None, binary_output_mrc_path = None):
    """
    loads a trained neural network, reads a cryo-EM map, runs the neural network on the cryo-EM map, applies a cutoff
    to the resulted map in order to round the values to either 0 or 1, and saves the original resulted map and
     he masked map to .mrc files.
    :param net_path: path of the trained neural network
    :param protein_mrc_path: path of the input cryo-EM map
    :param prob_output_mrc_path: path to save the .mrc prediction
    :param binary_output_mrc_path: path to save the .mrc binary mask of the prediction
    :return: the resulted helix prediction map
    """
    net = model.load_net(net_path)
    protein_data = torch.Tensor(dataset_manager.read_mrc(protein_mrc_path)).to('cuda')
    label_data = run_net_on_whole_protein(net, protein_data)
    actual_label_data_tensor = torch.Tensor(label_data).to('cuda')
    binary_label_data_tensor = dataset_manager.apply_cutoff(actual_label_data_tensor, 0.5)
    if prob_output_mrc_path != None:
        dataset_manager.save_mrc(label_data, prob_output_mrc_path)
    if binary_output_mrc_path != None:
        binary_label_data = binary_label_data_tensor.cpu().numpy()
        dataset_manager.save_mrc(binary_label_data, binary_output_mrc_path)
    # to calculate metrics uncomment the next line
    #metrics= validate_prediction(binary_label_data_tensor, dataset_manager.read_mrc(protein_mrc_path.split('.')[0] + '_helix.mrc')) # make it a comment later
    return label_data

def validate_prediction(prediction_labels, true_labels):
    """
        A function to validate the prediction, for the developer's use.
        The function prints the validation metrics (fp, tp, fn, tn, recall, precision, accuracy, f1
        :param prediction_labels: tensor of predicted helix mask of the protein
        :param true_labels: array of the true helix mask of the protein
        :return: dictionary of validation metrics
        """
    sz = dataset_manager.dimensions(prediction_labels)
    if sz != dataset_manager.dimensions(true_labels):
        print('dimensions of prediction are different from dimensions of true mask')
        return None
    true_label_data = torch.Tensor(true_labels).to('cuda')
    true_label_data = dataset_manager.apply_cutoff(true_label_data, 0.25)
    fp, tp, tn, fn = model.get_stats(prediction_labels, true_label_data)
    if tp == 0:
        precision = 0
        recall = 0
        f1 = 0
        accuracy = 0
    else:
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1 = 2 * precision * recall / (precision + recall)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    print(
        f'tn: {tn}, tp: {tp}, fn: {fn}, fp: {fp}, recall: {recall}, precision: {precision}, f1: {f1}, accuracy: {accuracy}')
    return {'tn': tn, 'tp': tp, 'fn': fn, 'fp': fp, 'recall': recall, 'precision': precision, 'f1': f1, 'accuracy': accuracy}