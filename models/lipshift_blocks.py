import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple

from .layers.conv import Conv2d
from .layers.activation import MinMax

class CenterNorm(nn.Module):
    r""" CenterNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, num_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

        self.scale = num_channels/(num_channels-1.0)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        x = self.scale*(x - u)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
    def lipschitz(self) -> torch.Tensor:
        return self.scale * self.weight.abs().max()
    
    def extra_repr(self) -> str:
        return f"scale={self.scale}"


class ScaleLayer(nn.Module):
    
    def __init__(self, alpha=0.1, learnable=True, dim=1):
        super().__init__()
        self.alpha = alpha
        self.learnable = learnable
        self.dim = dim
        if self.learnable:
            self.scale = nn.Parameter(torch.ones(dim) * self.alpha)
        else:
            self.scale = self.alpha

    def forward(self, x):
        if self.learnable:
            y = self.scale[:, None, None]*x
        else:
            y = self.scale*x
        return  y

    def lipschitz(self):
        if self.learnable:
            return self.scale.abs().max()
        
        return self.alpha
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}," \
               f"alpha={self.alpha}," \
               f"learnable={self.learnable}"


class LiResConv(nn.Module):

    def __init__(self,
                in_features,
                out_features=None,
                act_layer=MinMax,
                alpha=0.1):
        """ Linear Residual Convolution. By default is implemented by
        Conv2d layer. Our implementation uses the input format of [N, C, H, W].
        We use 3x3 convolution kernel.

        Args:
            in_features (int): input channels
            hidden_features (int): hidden channels, if None, set to in_features
            out_features (int): out channels, if None, set to in_features
            act_layer (callable): activation function class type
        """
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = Conv2d(in_features, out_features, 3)
        self.act = act_layer
        self.scale_layer = ScaleLayer(dim=in_features, alpha=alpha)  #**alpha_cfg)

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = shortcut + self.scale_layer(x)
        x = self.act(x)
        return x

    def lipschitz(self) -> torch.Tensor:
        lc = self.fc1.lipschitz()
        lc = lc * self.scale_layer.lipschitz()
        lc = (1+lc) * self.act.lipschitz()

        return lc


class LipShiFTBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_div=12,
                 drop_path=0.,
                 act_layer=MinMax,
                 norm_layer=CenterNorm,
                 input_resolution=None,
                 num_layers=12):
        """ The building block of LipShiFT architecture.

        Args:
            dim (int): feature dimension
            n_div (int): how many divisions are used. Totally, 4/n_div of
                channels will be shifted.
            drop_path (float): drop path prob.
            act_layer (callable): activation function class type.
            norm_layer (callable): normalization layer class type.
            input_resolution (tuple): input resolution. This optional variable
                is used to calculate the flops.

        """
        super(LipShiFTBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.drop_path_rate = drop_path
        self.num_layers = num_layers

        alpha = 0.2 #scaling factor
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = LiResConv(in_features=dim,
                        act_layer=act_layer,
                        alpha=alpha)
        self.scale_layer = ScaleLayer(dim=dim, alpha=alpha) 
        self.n_div = n_div

    def forward(self, x):
        x = self.shift_feat(x, self.n_div)
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.scale_layer(x)
        x = shortcut + self.drop_path(x)

        return x
    
    def lipschitz(self) -> torch.Tensor:
        lc = self.norm2.lipschitz()
        lc = lc * self.mlp.lipschitz() * self.scale_layer.lipschitz()

        if not self.training:
            keep_prob = 1. - self.drop_path_rate
            lc = 1. + (keep_prob * lc)
        else:
            lc = 1. + lc

        return lc

    def extra_repr(self) -> str:
        return f"dim={self.dim}," \
               f"input_resolution={self.input_resolution}," \
               f"shift percentage={4.0 / self.n_div * 100}%."

    @staticmethod
    def shift_feat(x, n_div):
        B, C, H, W = x.shape
        g = C // n_div
        out = torch.zeros_like(x)

        out[:, g * 0:g * 1, :, :-1] = x[:, g * 0:g * 1, :, 1:]  # shift left
        out[:, g * 1:g * 2, :, 1:] = x[:, g * 1:g * 2, :, :-1]  # shift right
        out[:, g * 2:g * 3, :-1, :] = x[:, g * 2:g * 3, 1:, :]  # shift up
        out[:, g * 3:g * 4, 1:, :] = x[:, g * 3:g * 4, :-1, :]  # shift down

        out[:, g * 4:, :, :] = x[:, g * 4:, :, :]  # no shift
        return out


class PatchMerging(nn.Module):

    def __init__(self, input_resolution, dim, norm_layer=CenterNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = Conv2d(dim, 2 * dim, kernel_size=2, stride=2, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.reduction(x)

        return x
    
    def lipschitz(self) -> torch.Tensor:
        lc = self.norm.lipschitz()
        lc = lc * self.reduction.lipschitz()

        return lc

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class BasicLayer(nn.Module):

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 n_div=12,
                 drop_path=None,
                 norm_layer=CenterNorm,
                 downsample=None,
                 use_checkpoint=False,
                 act_layer=MinMax,
                 num_layers=12):

        super(BasicLayer, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            LipShiFTBlock(dim=dim,
                          n_div=n_div,
                          drop_path=drop_path[i],
                          norm_layer=norm_layer,
                          act_layer=act_layer,
                          input_resolution=input_resolution,
                          num_layers=num_layers)
            for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution,
                                         dim=dim,
                                         norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x
    
    def lipschitz(self) -> torch.Tensor:
        lc = 1.0
        for blk in self.blocks:
            lc = lc * blk.lipschitz()

        if self.downsample is not None:
            lc = lc * self.downsample.lipschitz()

        return lc

    def extra_repr(self) -> str:
        return f"dim={self.dim}," \
               f"input_resolution={self.input_resolution}," \
               f"depth={self.depth}"


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int, tuple): Image size.
        patch_size (int, tuple): Patch token size.
        in_chans (int): Number of input image channels.
        embed_dim (int): Number of linear projection output channels.
        norm_layer (nn.Module, optional): Normalization layer.
    """

    def __init__(self,
                 img_size=32,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 norm_layer=CenterNorm):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0],
                              img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)

        return x
    
    def lipschitz(self) -> torch.Tensor:
        lc = self.proj.lipschitz()
        if self.norm is not None:
            lc = lc * self.norm.lipschitz()
        
        return lc
