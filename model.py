"""
Contains the convolutional neural net model and functions to train, evaluate, load and save the model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import dataset_manager
import random


def conv_block_3d(in_dim, out_dim, activation):
    """
    A 3D convolution followed by a 3D batch normalization and an activation function.
    :param in_dim: The number of input channels.
    :param out_dim: The number of output channels.
    :param activation: The activation function to be applied at the end.
    """
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation, )


def conv_trans_block_3d(in_dim, out_dim, activation):
    """
    Applies a 3D transposed convolution operator, a 3D batch normalization, and an activation function.
    :param in_dim: The number of input channels.
    :param out_dim: The number of output channels.
    :param activation: The activation function to be applied at the end.
    """
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        activation, )


def max_pooling_3d():
    """
    A max pooling layer.
    """
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def conv_block_2_3d(in_dim, out_dim, activation):
    """
    Applies conv_block_3d, followed by another 3D convolution and a batch normalization.
    :param in_dim:
    :param out_dim:
    :param activation: The activation function to be used in conv_block_3d
    """
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim), )


class UNet(nn.Module):
    """
    The 3D UNet model.
    """

    def __init__(self, in_dim, out_dim, num_filters):
        """
        Defines the structure of the model.
        """
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
        """
        Runs the model on an input.
        """
        # Down sampling
        down_1 = self.down_1(x)  # -> [1, 8, 32, 32, 32]
        pool_1 = self.pool_1(down_1)  # -> [1, 8, 16, 16, 16]

        down_2 = self.down_2(pool_1)  # -> [1, 16, 16, 16, 16]
        pool_2 = self.pool_2(down_2)  # -> [1, 16, 8, 8, 8]

        down_3 = self.down_3(pool_2)  # -> [1, 32, 8, 8, 8]
        pool_3 = self.pool_3(down_3)  # -> [1, 32, 4, 4, 4]

        down_4 = self.down_4(pool_3)  # -> [1, 64, 4, 4, 4]
        pool_4 = self.pool_4(down_4)  # -> [1, 64, 2, 2, 2]

        # Bridge
        bridge = self.bridge(pool_4)  # -> [1, 64, 1, 1, 1]

        # Up sampling

        trans_1 = self.trans_1(bridge)  # -> [1, 128, 4, 4, 4]
        concat_1 = torch.cat([trans_1, down_4], dim=1)  # -> [1, 192, 4, 4, 4]
        up_1 = self.up_1(concat_1)  # -> [1, 64, 4, 4, 4]

        trans_2 = self.trans_2(up_1)  # -> [1, 64, 8, 8, 8]
        concat_2 = torch.cat([trans_2, down_3], dim=1)  # -> [1, 96, 8, 8, 8]
        up_2 = self.up_2(concat_2)  # -> [1, 32, 8, 8, 8]

        trans_3 = self.trans_3(up_2)  # -> [1, 32, 16, 16, 16]
        concat_3 = torch.cat([trans_3, down_2], dim=1)  # -> [1, 48, 16, 16, 16]
        up_3 = self.up_3(concat_3)  # -> [1, 16, 16, 16, 16]

        trans_4 = self.trans_4(up_3)  # -> [1, 16, 32, 32, 32]
        concat_4 = torch.cat([trans_4, down_1], dim=1)  # -> [1, 24, 32, 32, 32]
        up_4 = self.up_4(concat_4)  # -> [1, 8, 32, 32, 32]

        # Output
        out = self.out(up_4)  # -> [1, 1, 32, 32, 32]
        return out


def get_5d_tensor(data):
    """
    1-liner to turn a numpy array into a 5D tensor of dimensions 1,1,32,32,32
    :param data: a numpy array
    :return: 5D tensor of the input data
    """
    return torch.Tensor(data).resize_(1, 1, 32, 32, 32)


def get_stats(output, label):
    """
    Calculates a confusion matrix (False Positive (FP), True Positive(TP)
    False Negative(FN), True Negative(TN)) given a probability map and its corresponding label
    :param output: a 3D probability map
    :param label: a 3D mask/label
    :return fp,tp,tn,tf as explained above
    """
    label.to('cuda')
    output.to('cuda')
    output_binary = torch.ceil(output-0.5)
    confusion_vector = output_binary / label
    tp = torch.sum(confusion_vector == 1).item()
    fp = torch.sum(confusion_vector == float('inf'))
    tn = torch.sum(torch.isnan(confusion_vector)).item()
    fn = torch.sum(confusion_vector == 0).item()
    return fp, tp, tn, fn


def run_net_on_patch(net, patch_data):
    """

    :param net:
    :param patch_data:
    :return:
    """
    patch_tensor = get_5d_tensor(patch_data)
    label_tensor = net(patch_tensor)
    return label_tensor[0, 0, :, :, :].detach().numpy()


def train_net(training_set, epochs=1, learning_rate=0.01):
    """
    Trains a new UNet on the training set on CUDA.
    :param training_set: The training set that was returned from split_training_validation_sets
    :param epochs: The number of epochs of the training loop
    :param learning_rate: The learning rate.
    """
    net = UNet(in_dim=1, out_dim=1, num_filters=8)
    net.to('cuda')
    criterion = nn.L1Loss().to('cuda')
    # create your optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        print('Shuffeling dataset')
        random.shuffle(training_set)
        for label, data in training_set:
            optimizer.zero_grad()  # zero the gradient buffers

            data_tensor = get_5d_tensor(data).to('cuda')
            label_tensor = get_5d_tensor(label).to('cuda')
            output = net(data_tensor)
            loss = criterion(output, label_tensor)
            loss.backward()
            optimizer.step()  # Does the update
    return net


def save_net(net, path):
    """
    Saves the net in the desired path
    :param net: the trained net
    :param path: the path where the trained net will be saved at
    :return
    """
    torch.save(net.state_dict(), path)


def load_net(path):
    """
    Loads a net from the desired path
    :param path: the path of the net
    :return net: the loaded net
    """
    net = UNet(in_dim=1, out_dim=1, num_filters=8)
    net.to('cuda')
    net.load_state_dict(torch.load(path))
    net.eval()
    return net


def print_and_write_to_file(text, file):
    print(text)
    if file:
        print(text, file=file)


def evaluate(net, validation_set, save_to=''):
    """
    The final stage of the training, evaluates the trained net using a validation set.
    The result will be written into a file with the following metrics:
    Accuracy, Recall and Precision
    :param net: the trained net
    :param validation_set: 30% of the input dataset, the set that will be evaluated on the trained net
    :param save_to: the path where the evaluation results will be written at
    :return
    """
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
        output_f_tensor = net(get_5d_tensor(data).to('cuda', non_blocking=True))
        output_array = output_f_tensor[0, 0, :, :, :]
        label.to('cuda', non_blocking=True)
        fp, tp, tn, fn = get_stats(output_array, label)
        fp_total += fp
        fn_total += fn
        tp_total += tp
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
    print_and_write_to_file('total false positives: ' + str(fp_total), file)
    print_and_write_to_file('total false negatives: ' + str(fn_total), file)
    print_and_write_to_file('total true positives: ' + str(tp_total), file)
    print_and_write_to_file('total true negatives: ' + str(tn_total), file)
    print_and_write_to_file(
        "recall: " + str(tp_total / (tp_total + fn_total)) + ", precision: " + str(tp_total / (tp_total + fp_total)),
        file)
    if save_to:
        file.close()
    if tp_total == 0:
        precision = 0
        recall = 0
    else:
        precision = tp_total / (tp_total + fp_total)
        recall = tp_total / (tp_total + fn_total)
    specificity = tn_total / (tn_total + fp_total)
    accuracy = (tp_total + tn_total) / (tp_total + tn_total + fp_total + fn_total)
    return {'fp': fp_total, 'fn': fn_total, 'tp': tp_total, 'tn': tn_total,
            'precision': precision, 'recall': recall, 'specificity': specificity, 'accuracy': accuracy}
