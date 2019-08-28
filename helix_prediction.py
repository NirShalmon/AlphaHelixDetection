import dataset_manager
import model
import numpy


def run_net_on_whole_protein(net, protein_data):
    size = dataset_manager.dimentions(protein_data)
    label_data = [[[0 for j in range(size[2])] for i in range(size[1])] for k in range(size[0])]
    for i in range(0,size[0], 32):
        for j in range(0, size[1], 32):
            for k in range(0, size[2] , 32):
                patch = dataset_manager.get_cube(protein_data, 32, i, j, k)
                labels = model.run_net_on_patch(net, patch)
                for ii in range(i, i+32):
                    for jj in range(j, j+32):
                        for kk in range(k, k+32):
                            if ii < size[0] and jj < size[1] and kk < size[2]:
                                label_data[ii][jj][kk]=labels[ii-i][jj-j][kk-k]
    return numpy.asarray(label_data)


def run_net_on_mrc_and_save(net_path, protein_mrc_path, output_mrc_path):
    net = model.load_net(net_path)
    protein_data = dataset_manager.read_mrc(protein_mrc_path)
    label_data = run_net_on_whole_protein(net, protein_data)
    dataset_manager.save_mrc(label_data, output_mrc_path)