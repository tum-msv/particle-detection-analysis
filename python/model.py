import torch
import torch.nn as nn
import math


class Conv1dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        """
        Class to perform 'same'-convolution
        Input:
        in_channels:    number of input channels
        out_channels:   number of output channels
        kernel_size:    kernel size
        stride:         stride to use in convolution (optional)
        dilation:      dilation to use in convolution (optional)
        """
        super().__init__()
        self.cut_last_element = (kernel_size % 2 == 0 and stride == 1 and dilation % 2 == 1)
        self.padding = math.ceil((1 - stride + dilation * (kernel_size-1))/2)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, stride=stride, dilation=dilation)

    def forward(self, x):
        if self.cut_last_element:
            return self.conv(x)[:, :, :-1]
        else:
            return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pad, norm, activation):
        """
        Convolutional building block consisting of convolution, batch-normalization and activation
        Input:
        in_channels:    number of input channels
        out_channels:   number of output channels
        kernel_size:    kernel size
        pad:            padding for convolution (either integer, 'same', etc.)
        norm:           bool to use batch-normalization or not
        activation:     activation to use ('relu' or 'tanh')
        """
        super(ConvBlock, self).__init__()
        if pad == 'same':
            self.conv = Conv1dSame(in_channels, out_channels, kernel_size)
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, bias=False)

        self.norm = norm
        self.bn = nn.BatchNorm1d(out_channels)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('Use valid activation function.')

    def forward(self, x):
        x = self.conv(x)

        if self.norm:
            x = self.bn(x)

        return self.activation(x)


class FCN(nn.Module):
    def __init__(self, channels, kernels, activation, n_labels):
        """
        Fully Convolutional Network (FCN) with Global Average Pooling at end
        Input:
        channels:       array that contains the intended amount of channels in one batch from begin to end
        kernels:        kernel size for every convolutional layer
        activation:     activation function to use
        no_classes:     total number of classes
        """
        super(FCN, self).__init__()
        self.no_blocks = len(channels)-1

        # list that contains all convolutional layers
        self.blocks = nn.ModuleList([ConvBlock(channels[i], channels[i+1], kernels[i],
                                               pad='same', norm=True, activation=activation)
                                     for i in range(self.no_blocks)])

        # cam layer, which is weighted sum of channels and as long as time series times number of labels
        self.cam_layer = nn.Linear(channels[-1], n_labels, bias=False)

        # self.output_layer = nn.Sequential(
        #     nn.Linear(in_features=channels[-1], out_features=n_labels, bias=True),
        #     nn.Softmax(dim=1)
        # )

    def forward(self, x):
        # apply all convolutional layers
        for block in self.blocks:
            x = block(x)

        # compute cam
        cam = self.cam_layer(x.transpose(1, 2))

        # perform global avg. pooling
        x = torch.mean(cam, dim=1, keepdim=False)

        assert ((len(x.shape) == 2) and (len(cam.shape) == 3))
        # return output of softmax
        return torch.log_softmax(x, dim=1), cam
