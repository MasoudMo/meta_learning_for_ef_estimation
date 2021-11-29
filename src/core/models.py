import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18


class Resnet2Plus1D(nn.Module):
    def __init__(self,
                 input_dim=128,
                 output_dim=128,
                 fc_dropout_p=0):
        """
        Constructor for the Resnet2Plus1D class
        :param output_dim: Dimension of output embedding
        :type output_dim: int
        :param fc_dropout_p: Dropout ratio used after each FC layer
        :type fc_dropout_p: float
        """

        super().__init__()

        # Initialize pretrained mode
        model = r2plus1d_18(pretrained=True)
        model.stem[0] = nn.Conv3d(1, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.r2p1model = nn.Sequential(*(list(model.children())[:-1]))

        # Linear transformation of output
        self.output_fc = nn.Linear(512, output_dim)

        # Other parameters
        self.fc_dropout_p = fc_dropout_p

    def forward(self, x):

        x = self.r2p1model(x.permute(0, 2, 1, 3, 4))

        # Flatten the features
        x = torch.flatten(x, start_dim=1, end_dim=-1)

        x = F.dropout(torch.tanh(self.output_fc(x)), p=self.fc_dropout_p, training=self.training)

        return x
