from matplotlib.pyplot import xkcd
import torch
import torch.nn as nn
import torch.functional as F

from typing import Tuple
from math import floor
from einops import rearrange


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
            x (torch.Tensor): Input tensor. Shape : b, c_in, h, w

        Returns:
            torch.Tensor: Output tensor. Shape : b, c_out, h, w
        """
        x = rearrange(x, "b c h w -> b h w c")
        x = self.fc_1(x)
        x = rearrange(x, "b h w c -> b c h w")
        x = self.act(x)
        x = rearrange(x, "b c h w -> b h w c")
        x = self.fc_2(x)
        x = rearrange(x, "b h w c -> b c h w")
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
            x (torch.Tensor): Input Tensor. Shape : b,  c_in, h, w

        Returns:
            torch.Tensor: Output Tensor. Shape : b, c_out, h, w
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


class Spatial_Proj(nn.Module):
    def __init__(self, in_channels: int, batch_size: int, out_channels: int = 0):
        """Initializes Spatial Proj block.

        Args:
            in_channels (int): Number of input channels
            out_channels (int, optional): Number of output channels. Defaults to 0.
            batch_size (int): Batch size.
        """
        super(Spatial_Proj, self).__init__()
        out_channels = out_channels or in_channels
        self.fc_1 = nn.Linear(out_channels, out_channels)
        self.cycle1 = CycleFC(
            in_channels=in_channels,
            out_channels=out_channels,
            sh=1,
            sw=7,
            batch_size=batch_size,
        )
        self.cycle2 = CycleFC(
            in_channels=in_channels,
            out_channels=out_channels,
            sh=1,
            sw=1,
            batch_size=batch_size,
        )
        self.cycle3 = CycleFC(
            in_channels=in_channels,
            out_channels=out_channels,
            sh=7,
            sw=1,
            batch_size=batch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for Spatial projection.

        Args:
            x (torch.Tensor): Input tensor. Shape: b,c,h,w

        Returns:
            torch.Tensor: Output tensor. Shape: b, c_out,h,w
        """
        x1 = self.cycle1(x)
        x2 = self.cycle2(x)
        x3 = self.cycle3(x)
        sum_x = x1 + x2 + x3
        sum_x = rearrange(sum_x, "b c h w -> b h w c")
        sum_x = self.fc_1(sum_x)
        sum_x = rearrange(sum_x, "b h w c -> b c h w")
        return sum_x


class MLP_Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 0,
        dropout: float = 0.0,
    ):
        """Initializes MLP block of cycleMLP

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            hidden_channels (int, optional): Number of hidden channels. Defaults to 0.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
        """
        super(MLP_Block, self).__init__()

        self.norm = nn.LayerNorm(in_channels)
        self.mlp = MLP(
            input_features=in_channels,
            hidden_features=hidden_channels,
            output_features=out_channels,
            drop=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for MLP block

        Args:
            x (torch.Tensor): Input tensor of shape b, c_in, h, w

        Returns:
            torch.Tensor: Output tensor of shape b, c_out, h, w
        """
        input_x = x
        x = rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        x = rearrange(x, "b h w c -> b c h w")
        x = self.mlp(x)
        x = x + input_x
        return x


class Cycle_Block(nn.Module):
    def __init__(self, in_channels: int, batch_size: int, out_channels: int = 0):
        """Initializes Cycle Block

        Args:
            in_channels (int): _description_
            batch_size (int): _description_
            out_channels (int, optional): _description_. Defaults to 0.
        """
        super(Cycle_Block, self).__init__()
        self.spatial_proj = Spatial_Proj(
            in_channels=in_channels, out_channels=out_channels, batch_size=batch_size
        )
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for cycle block

        Args:
            x (torch.Tensor): Input tensor with shape b,c_in,h,w.

        Returns:
            torch.Tensor: Output tensor with shape b,c_out,h,w.
        """
        input_x = x
        x = rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        x = rearrange(x, "b h w c -> b c h w")
        x = self.spatial_proj(x)
        x = x + input_x
        return x


class CycleMLP_Block(nn.Module):
    def __init__(self, in_channels: int, batch_size: int):
        """Initializes cycle mlp block

        Args:
            in_channels (int): Input channel.
            batch_size (int): Batch size.

        """
        super(CycleMLP_Block, self).__init__()
        self.cycle_block = Cycle_Block(
            in_channels=in_channels, batch_size=batch_size, out_channels=in_channels
        )
        self.mlp_block = MLP_Block(
            in_channels=in_channels,
            hidden_channels=4 * in_channels,
            out_channels=in_channels,
            dropout=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for CycleMLP_Block

        Args:
            x (torch.Tensor): Input tensor. shape: b,c,h,w

        Returns:
            torch.Tensor: Output tensor. shape: b,c,h,w
        """
        x = self.cycle_block(x)
        x = self.mlp_block(x)
        return x


if __name__ == "__main__":
    cfc = CycleMLP_Block(in_channels=3, batch_size=4)
    test_input = torch.randn(size=(4, 3, 64, 64))
    output = cfc(test_input)
    # print(output)
    print(output.shape)
