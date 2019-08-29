# 3D-UNet model.
import torch
import torch.nn as nn
import torch.optim as optim
import dataset_manager
import random
import numpy


def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation, )


def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        activation, )


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def conv_block_2_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim), )


class UNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(UNet, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.2, inplace=True)

        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_3 = max_pooling_3d()
        self.down_4 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)
        self.pool_4 = max_pooling_3d()

        # Bridge
        self.bridge = conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation)

        # Up sampling
        self.trans_1 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, activation)
        self.up_1 = conv_block_2_3d(self.num_filters * 24, self.num_filters * 8, activation)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
        self.up_2 = conv_block_2_3d(self.num_filters * 12, self.num_filters * 4, activation)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
        self.up_3 = conv_block_2_3d(self.num_filters * 6, self.num_filters * 2, activation)
        self.trans_4 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation)
        self.up_4 = conv_block_2_3d(self.num_filters * 3, self.num_filters * 1, activation)

        # Output
        self.out = conv_block_3d(self.num_filters, out_dim, activation)

    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x)  # -> [1, 4, 128, 128, 128]
        pool_1 = self.pool_1(down_1)  # -> [1, 4, 64, 64, 64]

        down_2 = self.down_2(pool_1)  # -> [1, 8, 64, 64, 64]
        pool_2 = self.pool_2(down_2)  # -> [1, 8, 32, 32, 32]

        down_3 = self.down_3(pool_2)  # -> [1, 16, 32, 32, 32]
        pool_3 = self.pool_3(down_3)  # -> [1, 16, 16, 16, 16]

        down_4 = self.down_4(pool_3)  # -> [1, 32, 16, 16, 16]
        pool_4 = self.pool_4(down_4)  # -> [1, 32, 8, 8, 8]


        # Bridge
        bridge = self.bridge(pool_4)  # -> [1, 128, 4, 4, 4]

        # Up sampling

        trans_1 = self.trans_1(bridge)  # -> [1, 64, 16, 16, 16]
        concat_1 = torch.cat([trans_1, down_4], dim=1)  # -> [1, 96, 16, 16, 16]
        up_1 = self.up_1(concat_1)  # -> [1, 32, 16, 16, 16]

        trans_2 = self.trans_2(up_1)  # -> [1, 32, 32, 32, 32]
        concat_2 = torch.cat([trans_2, down_3], dim=1)  # -> [1, 48, 32, 32, 32]
        up_2 = self.up_2(concat_2)  # -> [1, 16, 32, 32, 32]

        trans_3 = self.trans_3(up_2)  # -> [1, 16, 64, 64, 64]
        concat_3 = torch.cat([trans_3, down_2], dim=1)  # -> [1, 24, 64, 64, 64]
        up_3 = self.up_3(concat_3)  # -> [1, 8, 64, 64, 64]

        trans_4 = self.trans_4(up_3)  # -> [1, 8, 128, 128, 128]
        concat_4 = torch.cat([trans_4, down_1], dim=1)  # -> [1, 12, 128, 128, 128]
        up_4 = self.up_4(concat_4)  # -> [1, 4, 128, 128, 128]

        # Output
        out = self.out(up_4)  # -> [1, 3, 128, 128, 128]
        return out


def get_5d_tensor(data):
    output = torch.empty(1, 1, 32, 32, 32)
    for i in range(32):
        for j in range(32):
            for k in range(32):
                output[0][0][i][j][k] = data[i][j][k]
    return output


def get_false_positives(output, label):
    counter = 0
    for i in range(32):
        for j in range(32):
            for k in range(32):
                if output[i][j][k] > 0.5 and label[i][j][k] == 0:
                    counter += 1
    return counter


def get_false_negatives(output, label):
    counter = 0
    for i in range(32):
        for j in range(32):
            for k in range(32):
                if output[i][j][k] < 0.5 and label[i][j][k] == 1:
                    counter += 1
    return counter


def get_true_negatives(output, label):
    counter = 0
    for i in range(32):
        for j in range(32):
            for k in range(32):
                if output[i][j][k] < 0.5 and label[i][j][k] == 0:
                    counter += 1
    return counter


def get_true_positives(output, label):
    counter = 0
    for i in range(32):
        for j in range(32):
            for k in range(32):
                if output[i][j][k] > 0.5 and label[i][j][k] == 1:
                    counter += 1
    return counter


def run_net_on_patch(net, patch_data):
    patch_tensor = get_5d_tensor(patch_data)
    label_tensor = net(patch_tensor)
    return label_tensor[0, 0, :, :, :].detach().numpy()


def save_net(net, path):
    torch.save(net.state_dict(), path)


def load_net(path):
    net = UNet(in_dim=1, out_dim=1, num_filters=8)
    net.load_state_dict(torch.load(path))
    net.eval()
    return net


def print_and_write_to_file(text, file):
    print(text)
    if file:
        print(text, file=file)


def evaluate(net, validation_set, save_to=''):
    print('evaluation:')
    fp_total = 0
    fn_total = 0
    tp_total = 0
    tn_total = 0
    net.eval()  # set to evaluation mode
    if save_to:
        file = open(save_to, "w+")
    else:
        file = None
    for data, label in validation_set:
        output_f_tensor = net(get_5d_tensor(data))
        output_array = output_f_tensor[0, 0, :, :, :]
        fp = get_false_positives(output_array, label)
        fp_total += fp
        fn = get_false_negatives(output_array, label)
        fn_total += fn
        tp = get_true_positives(output_array, label)
        tp_total += tp
        tn = get_true_negatives(output_array, label)
        tn_total += tn
        print_and_write_to_file('false positives:' + str(fp), file)
        print_and_write_to_file('false negatives:' + str(fn), file)
        print_and_write_to_file('true positives:' + str(tp), file)
        print_and_write_to_file('true negatives:' + str(tn), file)
        if tp == 0:
            print_and_write_to_file("No true positives", file)
        else:
            print_and_write_to_file("recall: " + str(tp / (tp + fn)) + ", precision: " + str(tp / (tp + fp)), file)
        print_and_write_to_file('', file)
    print_and_write_to_file('total false positives: ' + str(fp_total),file)
    print_and_write_to_file('total false negatives: ' + str(fn_total),file)
    print_and_write_to_file('total true positives: ' + str(tp_total),file)
    print_and_write_to_file('total true negatives: ' + str(tn_total),file)
    print_and_write_to_file("recall: " + str(tp_total / (tp_total + fn_total)) + ", precision: " + str(tp_total / (tp_total + fp_total)), file)
    if save_to:
        file.close()
    if tp_total == 0:
        precision = 0
        recall = 0
    else:
        precision = tp_total/(tp_total + fp_total)
        recall = tp_total/(tp_total + fn_total)
    specificity = tn_total/(tn_total + fp_total)
    accuracy = (tp_total+tn_total)/(tp_total+tn_total+fp_total+fn_total)
    return {'fp': fp_total, 'fn': fn_total, 'tp': tp_total, 'tn': tn_total,
            'precision': precision, 'recall': recall, 'specificity': specificity, 'accuracy': accuracy}


def train_net(training_set, epochs=1, learning_rate=0.01):
    net = UNet(in_dim=1, out_dim=1, num_filters=8)
    criterion = nn.KLDivLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        print('Shuffeling dataset')
        random.shuffle(training_set)
        step_number = 0
        for data, label in training_set:
            optimizer.zero_grad()  # zero the gradient buffers

            data_tensor = get_5d_tensor(data)
            label_tensor = get_5d_tensor(label)
            output = net(data_tensor)

            loss = criterion(output, label_tensor)
            loss.backward()
            optimizer.step()  # Does the update
            print(f'step {step_number} out of {len(training_set)} of epoch {epoch} completed. loss: {loss}')
            step_number += 1
    return net


if __name__ == "__main__":
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 32
    print('getting dataset')
    dataset = list(dataset_manager.get_dataset())
    dataset = dataset[:20]
    training_set, validation_set = dataset_manager.split_training_validation_sets(dataset)
    net = train_net(training_set, epochs=0, learning_rate=0.01)
    print(evaluate(net, validation_set))
'''
    # dataset_manager.save_mrc(output_f_tensor[0,0,:,:30,:26].detach().numpy(), 'output_test.mrc'
