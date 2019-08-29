import dataset_manager
import model
import numpy




def run_net_on_whole_protein(net, protein_data):
    size = dataset_manager.dimentions(protein_data)
    label_data = [[[0 for j in range(size[2])] for i in range(size[1])] for k in range(size[0])]
    distance_from_cube_center = [[[10000000 for j in range(size[2])] for i in range(size[1])] for k in range(size[0])]
    for i in range(0, size[0]-16, 29):
        for j in range(0, size[1]-16, 29):
            for k in range(0, size[2]-16, 29):
                patch = dataset_manager.get_cube(protein_data, 32, i, j, k)
                labels = model.run_net_on_patch(net, patch)
                for ii in range(i, i + 32):
                    for jj in range(j, j + 32):
                        for kk in range(k, k + 32):
                            if ii < size[0] and jj < size[1] and kk < size[2]:
                                new_distance = abs(15.5-ii+i) + abs(15.5-jj+j) + abs(15.5-kk+k)
                                if new_distance < distance_from_cube_center[ii][jj][kk]:
                                    label_data[ii][jj][kk] = labels[ii - i][jj - j][kk - k]
                                    distance_from_cube_center[ii][jj][kk] = new_distance

    return numpy.asarray(label_data).astype('float32')


def calc_stats(output, label):
    fp=0
    fn=0
    tp=0
    tn=0
    size = dataset_manager.dimentions(label)
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
    net = model.load_net(net_path)
    protein_data = dataset_manager.read_mrc(protein_mrc_path)
    label_data = run_net_on_whole_protein(net, protein_data)
    true_label_data = dataset_manager.read_mrc(protein_mrc_path.split('.')[0] + '_helix.mrc')
    dataset_manager.apply_cutoff(true_label_data,0.25)
    calc_stats(label_data, true_label_data)
    dataset_manager.save_mrc(label_data, output_mrc_path)
