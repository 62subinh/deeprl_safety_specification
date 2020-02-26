import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lyapunov_reachability.common.utils import output_shape, View, ResBlock


def functional_finder(name):
    name = name.lower()
    if name == 'none':
        return no_act
    elif name == 'relu':
        return torch.relu
    elif name == 'tanh':
        return torch.tanh
    else:
        raise AttributeError('No activation is named {}'.format(name))


def activ_finder(name):
    name = name.lower()
    if name == 'relu':
        return nn.ReLU
    elif name == 'tanh':
        return nn.Tanh
    elif name == 'softsign':
        return nn.Softsign
    else:
        raise AttributeError('No activation is named {}'.format(name))


def no_act(input):
    return input


class Mlp(nn.Module):
    def __init__(self, in_, channels_=None, activ='relu', is_terminal=False):
        """
        :param in_      : Shape of the input (batch_size not included)
        :param channels_: Number of output channels for hidden layers in order.
        :param activ    : Name of the activation function; converted to nn.Module class via activ_finder.
        """
        super(Mlp, self).__init__()
        if channels_ is None:
            raise ValueError("Unable to create a network! Not enough parameters provided.")

        self.in_ = np.prod(np.array(in_), dtype=np.int)
        self.activ = activ_finder(activ)

        ops = [nn.Linear(self.in_, channels_[0],)]
        for i in range(1, len(channels_)):
            ops.append(self.activ())
            ops.append(nn.Linear(channels_[i-1], channels_[i],))
        if not is_terminal:
            ops.append(self.activ())
        self.ops = nn.Sequential(*ops)

    def forward(self, x):
        out = x.view(-1, self.in_)
        return self.ops(out)

    @classmethod
    def __name__(cls):
        return "Mlp"


class Cnn(nn.Module):
    def __init__(self, in_, channels_=None, kernel_sizes_=None, strides_=None, activ='relu', batch_norm=False,):
        """
        :param in_          : Shape of the 3-dimensional input.
        :param channels_    : Number of output channels for hidden layers in order.
        :param kernel_sizes_: Size of kernels for hidden layers in order.
        :param strides_     : Size of strides for hidden layers in order.
        :param activ        : Name of the activation function; converted to nn.Module class via activ_finder.
        :param batch_norm   : Boolean. Use batch normalization if True.
        """
        super(Cnn, self).__init__()
        if channels_ is None:
            raise ValueError("Unable to create a network! Not enough parameters provided.")

        self.activ = activ_finder(activ)
        if kernel_sizes_ is None:
            kernel_sizes_ = np.ones((len(channels_),), dtype=np.int) * 3
        if strides_ is None:
            strides_ = np.ones((len(channels_),), dtype=np.int) * 1
        ops = []
        channels_[0:0] = [in_[0]]
        for i in range(len(kernel_sizes_)):
            ops.append(nn.Conv2d(in_channels=channels_[i], out_channels=channels_[i+1],
                                 kernel_size=kernel_sizes_[i], stride=strides_[i], bias=True))
            if batch_norm:
                ops.append(nn.BatchNorm2d(channels_[i+1]))
            ops.append(self.activ())
        del channels_[0]
        self.ops = nn.Sequential(*ops)

    def forward(self, x):
        return self.ops(x)

    @classmethod
    def __name__(cls):
        return "Cnn"


class SumUpCnn(nn.Module):
    # TODO: add layer norm
    def __init__(self, in_, size_, layers, activ=nn.ReLU, name='sumup_cnn'):
        super(SumUpCnn, self).__init__()
        self.name = name
        self.activ = activ
        self.size_ = size_
        ops = []
        layers = [in_] + layers
        for i in range(len(layers)-1):
            ops.append(nn.Conv2d(layers[i], layers[i+1], kernel_size=3, stride=1, bias=True))
            ops.append(self.activ())
        self.ops = nn.Sequential(*ops)
        self.conv_final = nn.Conv2d(layers[-1], 1, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        out = self.ops(x)
        return self.conv_final(out).view(-1, self.size_).sum(-1)


class PooledCnn(nn.Module):
    # TODO: add layer_norm
    def __init__(self, in_, n_channel_, activ=nn.ReLU, name='pooled_cnn'):
        super(PooledCnn, self).__init__()
        self.name = name
        self.activ = activ

        self.first_ops = nn.Sequential(
            nn.Conv2d(in_channels=in_, out_channels=n_channel_, kernel_size=7, stride=1, bias=True),
            self.activ()
        )

        self.res_1 = ResBlock(n_channel_, name=name+'.res1')
        self.res_2 = ResBlock(n_channel_, name=name+'.res2')
        self.res_3 = ResBlock(n_channel_, name=name+'.res3')

        self.final_ops = nn.Sequential(
            nn.Conv2d(in_channels=n_channel_+in_, out_channels=n_channel_, kernel_size=1, stride=1, bias=True),
            self.activ()
        )

    def forward(self, x):
        feature = self.first_ops(x)
        feature = self.res_1(feature)
        feature = self.res_2(feature)
        feature = self.res_3(feature)
        # global_feature = self.temp_linear(feature.view(-1, 24*17*64))
        # global_feature = global_feature.view(-1, 64, 1, 1) * torch.ones(feature.size(), dtype=feature.dtype, device=feature.device)
        global_feature = F.adaptive_avg_pool2d(feature, (1, 1)) *\
                         torch.ones(feature.size(), dtype=feature.dtype, device=feature.device)
        return self.final_ops(torch.cat((x, global_feature), dim=1))

