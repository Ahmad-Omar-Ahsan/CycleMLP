import torch
import torch.nn as nn
import torch.functional as F

from typing import Tuple
from math import floor


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class MLP(nn.Module):
    def __init__(
        self,
        input_features: int,
        hidden_features: int = 0,
        output_features: int = 0,
        drop=0.0,
    ):
        """Initializes  MLP layer

        Args:
            input_features (int): Number of input features
            hidden_features (int, optional): Number of features of intermediate layer. Defaults to 0.
            output_features (int, optional): Number of outpput features Defaults to 0.
            drop (float, optional): Dropout probability . Defaults to 0.0.
        """
        super(MLP, self).__init__()
        output_features = output_features or input_features
        hidden_features = hidden_features or input_features
        self.fc_1 = nn.Linear(in_features=input_features, out_features=hidden_features)
        self.act = nn.GELU()
        self.fc_2 = nn.Linear(in_features=hidden_features, out_features=output_features)
        self.dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for MLP

        Args:
            x (torch.Tensor): Input tensor. Shape : b, h, w, c

        Returns:
            torch.Tensor: Output tensor. Shape : b, h, w, c_out
        """
        x = self.fc_1(x)
        x = self.act(x)
        x = self.fc_2(x)
        x = self.dropout(x)
        return x


class CycleFC(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, sh: int, sw: int, batch_size: int
    ):
        """Initialize CycleFC layer

        Args:
            in_channels (int): Input_channels.
            out_channels (int): Ouput_channels.
            sh (int): Stepsize along height.
            sw (int): Stepsize along width.
        """
        super(CycleFC, self).__init__()
        self.W_mlp = nn.parameter.Parameter(
            torch.randn(size=(batch_size, in_channels, out_channels)),
            requires_grad=True,
        )
        self.bias = nn.parameter.Parameter(torch.randn(size=(out_channels,)))

        self.sh = sh
        self.sw = sw
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for CycleFC

        Args:
            x (torch.Tensor): Input Tensor. Shape : b, h, w, c_in

        Returns:
            torch.Tensor: Output Tensor. Shape : b, h, w, c_out
        """
        b, c_in, h, w = x.shape
        output = torch.zeros(size=(b, self.out_channels, h, w))
        for i in range(h):
            for j in range(w):
                output[:, :, i, j] = self.calc_value(
                    x, w_mlp=self.W_mlp, c_in=c_in, i=i, j=j, h=h, w=w
                )
                output[:, :, i, j] += self.bias
        return output

    def get_offset(self, sh: int, sw: int, c: int) -> Tuple[int, int]:
        """Calculate offset based on stepsize along height and width for a given channel.

        Args:
            sh (int): Stepsize along height
            sw (int): Stepsize along width
            c (int): Current channel

        Returns:
            Tuple[int, int]: Returns offset along height and width for a given channel.
        """
        delta_i = c % sh - 1
        delta_j = floor(c / sh) % sw - 1
        return delta_i, delta_j

    def calc_value(
        self,
        x: torch.Tensor,
        w_mlp: torch.Tensor,
        c_in: int,
        i: int,
        j: int,
        h: int,
        w: int,
    ):
        """Calculate value for CycleFC

        Args:
            x (torch.Tensor): Input Tensor. Shape : b, c, h, w,
            w_mlp (torch.Tensor): Weight matrix.
            c_in (int): Input Channel
            i (int) : Current Height index
            j(int) : Current width index
            h(int) : Input height
            w(int) : Input weight
        Returns:
            torch.Tensor: Output tensor
        """
        sum = 0
        for c in range(c_in):
            delta_i, delta_j = self.get_offset(self.sh, self.sw, c)
            i_offset = (i + delta_i) % h
            j_offset = (j + delta_j) % w
            sum += torch.matmul(x[:, c, i_offset, j_offset], w_mlp[:, c, :])
        return sum


if __name__ == "__main__":
    cfc = CycleFC(in_channels=3, out_channels=12, sh=7, sw=1, batch_size=4)
    test_input = torch.randn(size=(4, 3, 64, 64))
    output = cfc(test_input)
    # print(output)
    # print(output.shape)
