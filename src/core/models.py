import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18
from math import floor


class Resnet2Plus1D(nn.Module):
    def __init__(self,
                 output_dim=128,
                 fc_dropout_p=0):

        super().__init__()

        # Initialize pretrained mode
        model = r2plus1d_18(pretrained=True)
        model.stem[0] = nn.Conv3d(1, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.r2p1model = nn.Sequential(*(list(model.children())[:-1]))

        # Linear transformation of output
        self.output_fc = nn.Linear(512, output_dim)

        # Other parameters
        self.fc_dropout_p = fc_dropout_p

    def forward(self,
                x):

        x = self.r2p1model(x.permute(0, 2, 1, 3, 4))

        # Flatten the features
        x = torch.flatten(x, start_dim=1, end_dim=-1)

        x = F.dropout(torch.tanh(self.output_fc(x)), p=self.fc_dropout_p, training=self.training)

        return x


class CustomCNN3D(nn.Module):
    """
    Custom 3D CNN used to generate vectors for input images
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 n_conv_layers=2,
                 out_channels=None,
                 kernel_sizes=None,
                 pool_sizes=None,
                 cnn_dropout_p=0,
                 fc_dropout_p=0):

        super().__init__()

        # Default list arguments
        if out_channels is None:
            out_channels = [4]*n_conv_layers
        if kernel_sizes is None:
            kernel_sizes = [3]*n_conv_layers
        if pool_sizes is None:
            pool_sizes = [2]*n_conv_layers

        # Ensure input params are list
        if type(out_channels) is not list:
            out_channels = [out_channels]*n_conv_layers
        else:
            assert len(out_channels) == n_conv_layers, 'Provide channel parameter for all layers.'
        if type(kernel_sizes) is not list:
            kernel_sizes = [kernel_sizes]*n_conv_layers
        else:
            assert len(kernel_sizes) == n_conv_layers, 'Provide kernel size parameter for all layers.'
        if type(pool_sizes) is not list:
            pool_sizes = [pool_sizes]*n_conv_layers
        else:
            assert len(pool_sizes) == n_conv_layers, 'Provide pool size parameter for all layers.'

        # Conv layers
        convs = list()

        # Add first layer
        convs.append(nn.Sequential(nn.Conv3d(in_channels=1,
                                             out_channels=out_channels[0],
                                             kernel_size=kernel_sizes[0]),
                                   nn.BatchNorm3d(out_channels[0]),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool3d(kernel_size=(pool_sizes[0], pool_sizes[0], pool_sizes[0])),
                                   nn.Dropout3d(p=cnn_dropout_p)))

        # Add subsequent layers
        for layer_num in range(1, n_conv_layers):
            convs.append(nn.Sequential(nn.Conv3d(in_channels=out_channels[layer_num-1],
                                                 out_channels=out_channels[layer_num],
                                                 kernel_size=kernel_sizes[layer_num]),
                                       nn.BatchNorm3d(out_channels[layer_num]),
                                       nn.ReLU(inplace=True),
                                       nn.MaxPool3d(kernel_size=(pool_sizes[layer_num],
                                                                 pool_sizes[layer_num],
                                                                 pool_sizes[layer_num])),
                                       nn.Dropout3d(p=cnn_dropout_p)))

        # Add adaptive pooling to last layer
        convs[-1].add_module(name='adaptive_pool', module=nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)))
        self.conv = nn.Sequential(*convs)

        # Output linear layer
        self.output_fc = nn.Sequential(nn.Linear(out_channels[-1], output_dim),
                                       nn.Dropout(p=fc_dropout_p),
                                       nn.Tanh())

    def forward(self,
                x):

        x = x.permute(0, 2, 1, 3, 4)

        # CNN layers
        x = self.conv(x)

        # Flatten the features
        x = torch.flatten(x, start_dim=1, end_dim=-1)

        # Output FC layer
        x = self.output_fc(x)

        return x


class CustomCNN3DSpatial(nn.Module):
    """
    Custom 3D CNN used to generate vectors for input images
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 n_conv_layers=2,
                 out_channels=None,
                 kernel_sizes=None,
                 pool_sizes=None,
                 cnn_dropout_p=0,
                 fc_dropout_p=0):

        super().__init__()

        # Default list arguments
        if out_channels is None:
            out_channels = [4]*n_conv_layers
        if kernel_sizes is None:
            kernel_sizes = [3]*n_conv_layers
        if pool_sizes is None:
            pool_sizes = [2]*n_conv_layers

        # Ensure input params are list
        if type(out_channels) is not list:
            out_channels = [out_channels]*n_conv_layers
        else:
            assert len(out_channels) == n_conv_layers, 'Provide channel parameter for all layers.'
        if type(kernel_sizes) is not list:
            kernel_sizes = [kernel_sizes]*n_conv_layers
        else:
            assert len(kernel_sizes) == n_conv_layers, 'Provide kernel size parameter for all layers.'
        if type(pool_sizes) is not list:
            pool_sizes = [pool_sizes]*n_conv_layers
        else:
            assert len(pool_sizes) == n_conv_layers, 'Provide pool size parameter for all layers.'

        # Compute paddings to preserve temporal dim
        paddings = list()
        for kernel_size in kernel_sizes:
            paddings.append(floor((kernel_size - 1) / 2))

        # Conv layers
        convs = list()

        # Add first layer
        convs.append(nn.Sequential(nn.Conv3d(in_channels=1,
                                             out_channels=out_channels[0],
                                             kernel_size=kernel_sizes[0],
                                             padding=(paddings[0], 0, 0)),
                                   nn.BatchNorm3d(out_channels[0]),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool3d(kernel_size=(1, pool_sizes[0], pool_sizes[0])),
                                   nn.Dropout3d(p=cnn_dropout_p)))

        # Add subsequent layers
        for layer_num in range(1, n_conv_layers):
            convs.append(nn.Sequential(nn.Conv3d(in_channels=out_channels[layer_num-1],
                                                 out_channels=out_channels[layer_num],
                                                 kernel_size=kernel_sizes[layer_num],
                                                 padding=(paddings[0], 0, 0)),
                                       nn.BatchNorm3d(out_channels[layer_num]),
                                       nn.ReLU(inplace=True),
                                       nn.MaxPool3d(kernel_size=(1, pool_sizes[layer_num], pool_sizes[layer_num])),
                                       nn.Dropout3d(p=cnn_dropout_p)))

        # Add adaptive pooling to last layer
        convs[-1].add_module(name='adaptive_pool', module=nn.AdaptiveAvgPool3d(output_size=(None, 1, 1)))
        self.conv = nn.Sequential(*convs)

        # Output linear layer
        self.output_fc = nn.Sequential(nn.Linear(out_channels[-1], output_dim),
                                       nn.Dropout(p=fc_dropout_p),
                                       nn.Tanh())

    def forward(self,
                x):

        x = x.permute(0, 2, 1, 3, 4)

        # CNN layers
        x = self.conv(x).permute(0, 2, 1, 3, 4)

        # Flatten the features
        x = torch.flatten(x, start_dim=2, end_dim=-1)

        # Output FC layer
        x = self.output_fc(x)

        return x
