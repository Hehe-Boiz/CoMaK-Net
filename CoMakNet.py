from torch.nn import functional as F
from torch import nn
from utils.utils import h_swish, _make_divisible, SELayer, ECALayer, merge_pre_bn, PatchEmbed
import natten
from natten import NeighborhoodAttention2D as NeighborhoodAttention
from functools import partial
from timm.models.layers import DropPath, trunc_normal_
import torch
from utils.utils import NORM_EPS, Mamba2DBlock

is_natten_post_017 = hasattr(natten, "context")

class LocalityFeedForward(nn.Module):
    def __init__(self, in_dim=64, out_dim=96, kernel_size=3, stride=1, expand_ratio=4., act='hs+se', reduction=4,
                 wo_dp_conv=False, dp_first=False):
        """
        :param in_dim: the input dimension
        :param out_dim: the output dimension. The input and output dimension should be the same.
        :param stride: stride of the depth-wise convolution.
        :param expand_ratio: expansion ratio of the hidden dimension.
        :param act: the activation function.
                    relu: ReLU
                    hs: h_swish
                    hs+se: h_swish and SE module
                    hs+eca: h_swish and ECA module
                    hs+ecah: h_swish and ECA module. Compared with eca, h_sigmoid is used.
        :param reduction: reduction rate in SE module.
        :param wo_dp_conv: without depth-wise convolution.
        :param dp_first: place depth-wise convolution as the first layer.
        """
        super(LocalityFeedForward, self).__init__()
        hidden_dim = int(in_dim * expand_ratio)


        layers = []
        # the first linear layer is replaced by 1x1 convolution.
        layers.extend([
            nn.Conv2d(in_dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)])

        # the depth-wise convolution between the two linear layers
        if not wo_dp_conv:
            dp = [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size= kernel_size, stride= stride, padding= kernel_size // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)
            ]
            if dp_first:
                layers = dp + layers
            else:
                layers.extend(dp)

        if act.find('+') >= 0:
            attn = act.split('+')[1]
            if attn == 'se':
                layers.append(SELayer(hidden_dim, reduction=reduction))
            elif attn.find('eca') >= 0:
                layers.append(ECALayer(hidden_dim, sigmoid=attn == 'eca'))
            else:
                raise NotImplementedError('Activation type {} is not implemented'.format(act))

        # the second linear layer is replaced by 1x1 convolution.
        layers.extend([
            nn.Conv2d(hidden_dim, out_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_dim)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = x + self.conv(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, out_features=None, mlp_ratio=None, drop=0., bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_dim = _make_divisible(in_features * mlp_ratio, 32)
        self.conv1 = nn.Conv2d(in_features, hidden_dim, kernel_size=1, bias=bias)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, out_features, kernel_size=1, bias=bias)
        self.drop = nn.Dropout(drop)

    def merge_bn(self, pre_norm):
        merge_pre_bn(self.conv1, pre_norm)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x


class LFP(nn.Module):
    """
    Efficient Convolution Block
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, path_dropout=0.2,
                 drop=0, head_dim=32, mlp_ratio=3):
        super(LFP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        assert out_channels % head_dim == 0

        self.patch_embed = PatchEmbed(in_channels, out_channels, stride)
        #self.mhca = MHCA(out_channels, head_dim)
        self.norm1 = norm_layer(out_channels)
        extra_args = {"rel_pos_bias": True} if is_natten_post_017 else {"bias": True}
        self.attn = NeighborhoodAttention(
            out_channels,
            kernel_size=7,
            dilation=None,
            num_heads= (out_channels // head_dim),
            qkv_bias=True,
            qk_scale=None,
            attn_drop=drop,
            proj_drop=0.0,
            **extra_args,
        )
        self.attention_path_dropout = DropPath(path_dropout)

        self.conv = LocalityFeedForward(out_channels, out_channels, kernel_size, 1, mlp_ratio, reduction=out_channels)

        self.norm2 = norm_layer(out_channels)
        #self.mlp = Mlp(out_channels, mlp_ratio=mlp_ratio, drop=drop, bias=True)
        #self.mlp_path_dropout = DropPath(path_dropout)
        #hidden_dim = int(out_channels * mlp_ratio)
        #self.kan = KAN([out_channels, hidden_dim, out_channels])
        self.is_bn_merged = False

    def merge_bn(self):
        if not self.is_bn_merged:
            self.mlp.merge_bn(self.norm)
            self.is_bn_merged = True

    def forward(self, x):
        x = self.patch_embed(x)
        b, c, h, w = x.shape
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x.reshape(b, h, w, c))
        x = shortcut + self.attention_path_dropout(x.reshape(b, c, h, w))
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm2(x)
        else:
            out = x
        #x = x + self.mlp_path_dropout(self.mlp(out))
        x = x + self.conv(out) # (B, dim, 14, 14)
        #b, d, t, _ = out.shape
        #x = x + self.mlp_path_dropout(self.kan(out.reshape(-1, out.shape[1])).reshape(b, d, t, t))
        return x

class ConvGating(nn.Module):
    def __init__(self, C, Cp):
        super().__init__()
        self.pw_proj = nn.Conv2d(C, Cp, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_proj = nn.BatchNorm2d(Cp)

        self.act_mid = nn.ReLU()

        self.pw_expand = nn.Conv2d(Cp, C, kernel_size=1, stride=1, padding=0, bias=False)
        self.act_gate = nn.Sigmoid()

    def forward(self, x):
        x = self.pw_proj(x)  # C → Cp
        x = self.bn_proj(x)
        x = self.act_mid(x)
        x = self.pw_expand(x)  # Cp → C
        x = self.act_gate(x)  # gating
        return x


class LinearGating(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        reduced_dim = max(dim // reduction, 1)
        self.gating = nn.Sequential(
            nn.Linear(dim, reduced_dim, bias=False),
            nn.LayerNorm(reduced_dim),
            nn.SiLU(),
            nn.Linear(reduced_dim, dim, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, H, W]

        # chuyển sang Channel Last [B, H, W, C] để dùng Linear/LN
        x_in = x.permute(0, 2, 3, 1)
        gate = self.gating(x_in)

        # trả về Channel First [B, C, H, W] để nhân
        return gate.permute(0, 3, 1, 2)

class LocalPath(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        expansion_ratio: int = 1, # > 1
        **kwargs,
    ):
        super().__init__()
        C = hidden_dim
        Ce = C * expansion_ratio

        self.pw_expand   = nn.Conv2d(C, Ce, kernel_size=1, stride=1, padding=0, bias=False)
        self.pw_proj     = nn.Conv2d(Ce, C, kernel_size=1, stride=1, padding=0, bias=False)
        self.dw     = nn.Conv2d(Ce, Ce, kernel_size=3, stride=1, padding=1, groups=Ce, bias=False)

        self.bn_in  = nn.BatchNorm2d(C)
        self.bn1    = nn.BatchNorm2d(Ce)
        self.bn2    = nn.BatchNorm2d(Ce)
        self.bn3    = nn.BatchNorm2d(C)

        self.act = nn.SiLU()

    def forward(self, x):
        # Channel First
        x = self.bn_in(x)

        x = self.pw_expand(x) # C -> Ce
        x = self.bn1(x)
        x = self.act(x)

        x = self.dw(x)  # C = Ce
        x = self.bn2(x)
        x = self.act(x)

        x = self.pw_proj(x) # Ce -> C
        x = self.bn3(x) # gate voi self.gating GlobalExtractor
        return x

def r_silu(x):
    return x * torch.sigmoid(-x)

class GlobalPath(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            expansion_ratio: int = 1,  # > 1
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
            **kwargs,
    ):
        super().__init__()
        C = hidden_dim
        Ce = int(C * expansion_ratio)

        self.norm = nn.LayerNorm(normalized_shape=C, eps=NORM_EPS)

        self.ln_ex = nn.Linear(in_features=C, out_features=2*Ce)
        self.ln_proj = nn.Linear(in_features=Ce, out_features=C)
        self.act  = nn.SiLU()

        self.dw   = nn.Conv2d(Ce, Ce, kernel_size=3, stride=1, padding=1, groups=Ce, bias=False)

        self.ssm  = Mamba2DBlock(d_inner=Ce).to(device)

    def forward(self, x: nn.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous() # (B, H, W, C)

        x = self.norm(x)
        uv = self.ln_ex(x)
        u, v = uv.chunk(2, dim=3)  # split theo channel last

        gate = r_silu(u)  # (B, H, W, Ce)

        g = self.act(u)
        v = v.permute(0, 3, 1, 2).contiguous() # (B, Ce, H, W)
        v = self.dw(v)
        v = self.act(v)
        v = v.permute(0, 2, 3, 1).contiguous() # (B, H, W, Ce)

        gate = v * gate

        v = self.ssm(v)
        v = v * g
        v = v + gate

        v = self.ln_proj(v) # gate voi self.gating cua LocalExtractor
        v = v.permute(0, 3, 1, 2).contiguous() # (B, Ce, H, W)

        return v

class EnhancedMamba2DBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            expansion_ratio_local: int = 1,
            expansion_ratio_global: int = 1,
            contraction_ratio_gating: int = 1,
            drop_path: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        self.norm = nn.BatchNorm2d(hidden_dim)
        self.global_path = GlobalPath(hidden_dim=hidden_dim, expansion_ratio=expansion_ratio_global)
        self.local_path = LocalPath(hidden_dim=hidden_dim, expansion_ratio=expansion_ratio_local)

        self.gating_local = LinearGating(dim=hidden_dim, reduction=contraction_ratio_gating)
        self.gating_global = LinearGating(dim=hidden_dim, reduction=contraction_ratio_gating)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x

        global_path = self.global_path(x)
        local_path = self.local_path(x)

        # cross gating
        gating_for_global = self.gating_local(local_path)
        gating_for_local = self.gating_global(global_path)

        x_fused = global_path * gating_for_global + local_path * gating_for_local

        # Residual Connection
        # Norm -> DropPath -> Add Shortcut
        x_out = shortcut + self.drop_path(self.norm(x_fused))
        return x_out
