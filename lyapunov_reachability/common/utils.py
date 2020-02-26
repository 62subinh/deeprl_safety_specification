import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import constraints
from torch.distributions.transforms import Transform


def conv2d_output_shape(size, kernel_size, stride, padding=0, dilation=1,):
    return (size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def output_shape(input_shape, params):
    channels_ = params.get('channels_')
    kernel_sizes_ = params.get('kernel_sizes_')
    strides_ = params.get('strides_')

    if kernel_sizes_ is None or strides_ is None:
        return (channels_[-1],)

    size = input_shape[1]
    for l in range(len(kernel_sizes_)):
        size = conv2d_output_shape(size, kernel_sizes_[l], strides_[l])
    return (channels_[-1], size, size)


def init_weights(model, conservative_init=False):
    initial_bias = float(conservative_init)
    for idx, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, initial_bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, initial_bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def clip_but_pass_gradient(input_, lower=-1., upper=1.):
    clip_up = input_ > upper
    clip_low = input_ < lower
    clipper = (upper - input_) * clip_up.to(dtype=torch.float32) + (lower - input_) * clip_low.to(dtype=torch.float32)
    return input_ + clipper.detach()


class View(nn.Module):
    def __init__(self, shape_):
        super(View, self).__init__()
        self.shape_ = shape_

    def forward(self, x):
        shape_ = (-1,) + self.shape_
        return x.view(*shape_)


class ResBlock(nn.Module):
    # TODO: add layer_norm
    def __init__(self, n_channel_, kernel_size_=3, activ_fn=F.relu, name='res_block'):
        super(ResBlock, self).__init__()
        self.name = name
        self.activ = activ_fn
        self.ops = nn.Sequential(
            nn.Conv2d(in_channels=n_channel_, out_channels=n_channel_, kernel_size=kernel_size_, stride=1, bias=True),
            nn.BatchNorm2d(n_channel_),
            nn.Conv2d(in_channels=n_channel_, out_channels=n_channel_, kernel_size=kernel_size_, stride=1, bias=True),
            nn.BatchNorm2d(n_channel_)
        )

    def forward(self, x):
        return self.activ(x + self.ops(x))
