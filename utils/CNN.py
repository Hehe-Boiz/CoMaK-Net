import torch.nn as nn
from timm.models.layers import DropPath

class CNNStem(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0.0,
        expansion_ratio: float = 0.0, # > 1
        projection_ratio: float = 0.0, # < 1
        **kwargs,
    ):
        super().__init__()
        C = hidden_dim
        Ce = int(C * expansion_ratio)
        Cp = int(C * projection_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.pw_expand_ex   = nn.Conv2d(C, Ce, kernel_size=1, stride=1, padding=0, bias=False)
        self.pw_proj_ex     = nn.Conv2d(Ce, C, kernel_size=1, stride=1, padding=0, bias=False)

        self.pw_proj_proj = nn.Conv2d(C, Cp, kernel_size=1, stride=1, padding=0, bias=False)
        self.pw_expand_proj = nn.Conv2d(Cp, C, kernel_size=1, stride=1, padding=0, bias=False)

        self.dw  = nn.Conv2d(Ce, Ce, kernel_size=3, stride=1, padding=1, groups=Ce, bias=False)

        self.bn_in = nn.BatchNorm2d(C)
        self.bn1 = nn.BatchNorm2d(Ce)
        self.bn2 = nn.BatchNorm2d(Ce)
        self.bn3 = nn.BatchNorm2d(C)
        self.bn4 = nn.BatchNorm2d(Cp)

        self.act_1 = nn.SiLU()
        self.act_2 = nn.ReLU()
        self.act_3 = nn.Sigmoid()

    def forward(self, x):
        x = self.bn_in(x)

        x = self.pw_expand_ex(x)
        x = self.bn1(x)
        x = self.act_1(x)
        x = self.dw(x)  # C = Ce
        x = self.bn2(x)
        x = self.act_1(x)
        x = self.pw_proj_ex(x)
        x = self.bn3(x)
        short_cut = x
        x = self.pw_proj_proj(x)
        x = self.bn4(x)
        x = self.act_2(x)
        x = self.pw_expand_proj(x)
        x = self.act_3(x)
        return x