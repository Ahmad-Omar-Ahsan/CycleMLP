import torch
import torch.nn as nn

from typing import Tuple
from math import floor
from einops import rearrange, reduce


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
        self,
        in_channels: int,
        out_channels: int,
        sh: int,
        sw: int,
        batch_size: int,
        device: str,
    ):
        """Initialize CycleFC layer

        Args:
            in_channels (int): Input_channels.
            out_channels (int): Ouput_channels.
            sh (int): Stepsize along height.
            sw (int): Stepsize along width.
            device (str) : Device
        """
        super(CycleFC, self).__init__()
        self.W_mlp = nn.parameter.Parameter(
            torch.randn(size=(batch_size, in_channels, out_channels)),
            requires_grad=True,
        ).to(device)
        self.bias = nn.parameter.Parameter(torch.randn(size=(out_channels,))).to(device)
        self.device = device
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
        output = torch.zeros(size=(b, self.out_channels, h, w)).to(self.device)
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
    def __init__(
        self,
        in_channels: int,
        batch_size: int,
        device: str,
        out_channels: int = 0,
    ):
        """Initializes Spatial Proj block.

        Args:
            in_channels (int): Number of input channels
            out_channels (int, optional): Number of output channels. Defaults to 0.
            batch_size (int): Batch size.
            device (str): Device type
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
            device=device,
        )
        self.cycle2 = CycleFC(
            in_channels=in_channels,
            out_channels=out_channels,
            sh=1,
            sw=1,
            batch_size=batch_size,
            device=device,
        )
        self.cycle3 = CycleFC(
            in_channels=in_channels,
            out_channels=out_channels,
            sh=7,
            sw=1,
            batch_size=batch_size,
            device=device,
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
    def __init__(
        self, in_channels: int, batch_size: int, device: str, out_channels: int = 0
    ):
        """Initializes Cycle Block

        Args:
            in_channels (int): _description_
            batch_size (int): _description_
            out_channels (int, optional): _description_. Defaults to 0.
            device (str) : Device type
        """
        super(Cycle_Block, self).__init__()
        self.spatial_proj = Spatial_Proj(
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            device=device,
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
    def __init__(self, in_channels: int, batch_size: int, device: str):
        """Initializes cycle mlp block

        Args:
            in_channels (int): Input channel.
            batch_size (int): Batch size.
            device (str) : device

        """
        super(CycleMLP_Block, self).__init__()
        self.cycle_block = Cycle_Block(
            in_channels=in_channels,
            batch_size=batch_size,
            out_channels=in_channels,
            device=device,
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


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        """Initialize overlap patch embedding

        Args:
            img_size (int, optional): Image size. Defaults to 224.
            patch_size (int, optional): Patch size. Defaults to 7.
            stride (int, optional): Stride. Defaults to 4.
            in_chans (int, optional): Input Channels. Defaults to 3.
            embed_dim (int, optional): Embed dim. Defaults to 768.
        """
        super().__init__()
        img_size = pair(img_size)
        patch_size = pair(patch_size)

        assert max(patch_size) > stride, "Set larger patch_size than stride"

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x


class Stage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stride: int,
        layers: int,
        output_channel: int,
        batch_size: int,
        device: str,
    ):
        """Initialize a stage in CycleMLP

        Args:
            in_channels (int): Number of input channels.
            stride (int): Strides.
            device (str): Device Type
            layers (int): Number of layers of CycleMLP_Block
            output_channel (int): Number of output channels
            batch_size (int): Batch size
        """
        super(Stage, self).__init__()
        self.patch = OverlapPatchEmbed(
            in_chans=in_channels, stride=stride, patch_size=7, embed_dim=output_channel
        )
        cycle_mlps = []

        for l in range(layers):
            cycle_mlps.append(
                CycleMLP_Block(
                    in_channels=output_channel, batch_size=batch_size, device=device
                )
            )
        self.CycleMLP_block = nn.Sequential(*cycle_mlps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for a stage

        Args:
            x (torch.Tensor): Input tensor of shape b,c_in,h,w

        Returns:
            torch.Tensor: Output tensor of shape  b,c_out,h/stride,w/stride
        """
        x = self.patch(x)
        x = self.CycleMLP_block(x)
        return x


class CycleMLP(nn.Module):
    def __init__(
        self,
        stride_list: list,
        channel_list: list,
        in_channels: list,
        layer_list: list,
        batch_size: int,
        device: str,
        num_class: int,
    ):
        """Initializes CycleMLP.

        Args:
            stride_list (list): List of stride values.
            channel_list (list): List of channel values.
            in_channels (list): List of input channel values
            device (str): Device type.
            layer_list (list): List of layers for CycleMLP block.
            batch_size (int) : Batch size
            num_class (int) : Number of classes
        """
        super(CycleMLP, self).__init__()
        stages = []
        list_range = len(stride_list)
        for s in range(list_range):
            stages.append(
                Stage(
                    in_channels=in_channels[s],
                    stride=stride_list[s],
                    layers=layer_list[s],
                    output_channel=channel_list[s],
                    batch_size=batch_size,
                    device=device,
                )
            )
        self.stages = nn.Sequential(*stages)
        self.to_logits = nn.Sequential(nn.Linear(channel_list[-1], num_class))
        self.norm = nn.LayerNorm(channel_list[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for CycleMLP

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor
        """
        x = self.stages(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.norm(x)
        x = self.to_logits(x.mean(1))
        return x


if __name__ == "__main__":
    # cfc = CycleMLP_Block(in_channels=3, batch_size=4)
    cmlp = CycleMLP(
        stride_list=[4, 2],
        channel_list=[96, 192],
        layer_list=[2, 2],
        batch_size=4,
        in_channels=[3, 96],
        device="cuda",
        num_class=10,
    ).to("cuda")
    test_input = torch.randn(size=(4, 3, 64, 64)).to("cuda")
    parameters = sum(p.numel() for p in cmlp.parameters())
    print(parameters)
    # output = cfc(test_input)
    # # print(output)
    # print(output.shape)
    # p = OverlapPatchEmbed(img_size=64,embed_dim=3, stride=4, patch_size=7)
    # d = OverlapPatchEmbed(embed_dim=192, stride=2, patch_size=7, in_chans=128)
    output = cmlp(test_input)
    # output = d(output)
    print(output.shape)
