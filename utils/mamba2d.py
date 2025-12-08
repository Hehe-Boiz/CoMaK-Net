import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from timm.layers import DropPath
from collections import OrderedDict

from kernels.wavefront_cuda import wavefront_scan_cuda
from .utils import StarReLU, SepConv
# Increase torch.compile cache size limit
torch._dynamo.config.cache_size_limit = 64


class Mamba2DBlock(nn.Module):
    def __init__(self,
                 d_inner: int,
                 d_state: int = 16, 
                 dt_rank: int | str = 'auto', # rank của ma trận factorization cho deltaT và deltaL
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 dt_init: str = "random",  
                 dt_scale: float = 1.0,
                 dt_init_floor=1e-4,
                 double_scans: bool = False):
        super().__init__()
        self.d_inner = d_inner # ED
        self.d_state = d_state # N
        self.double_scans = double_scans
        
        # rank for deltaT, detaL
        if dt_rank == "auto":
            dt_rank = math.ceil(d_inner/16)
        self.dt_rank = dt_rank

        # init dt
        dt = torch.exp(
                torch.rand(self.d_inner) 
                * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
                ).clamp(min=dt_init_floor) # rounding number
        
        # inverse softplus
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        
        # x -> [deltaT, deltaL, B, C]
        self.x_proj = nn.Linear(self.d_inner, 2*self.dt_rank + 2*self.d_state)

        # channel last
        # deltaT, deltaL: (B,H,W,dt_rank) -> (B, H, W, ED) 
     
        self.dt_projT = nn.Linear(self.dt_rank, self.d_inner)
        self.dt_projL = nn.Linear(self.dt_rank, self.d_inner)

        # init weight for dt_proj*
        if dt_init == "constant":
            nn.init.constant_(self.dt_projT.weight, dt_init_std)
            nn.init.constant_(self.dt_projL.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_projT.weight, -dt_init_std, dt_init_std)
            nn.init.uniform_(self.dt_projL.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        with torch.no_grad():# Turn off gradient tracking in this block 
            self.dt_projT.bias.copy_(inv_dt)
            self.dt_projL.bias.copy_(inv_dt)

        # Structured State Space Sequence Model - Diagonal
        # S4D-style init
        AT = torch.arange(1, self.d_state + 1,
                          dtype=torch.float32).repeat(self.d_inner, 1)
        AL = AT.detach().clone()
        self.AT_log = nn.Parameter(torch.log(AT)) # (ED, N)
        # Flag tùy chỉnh của tác giả
        # gán nhãn lên một tham số để báo cho optimizer
        # hoặc weight decay filter rằng tham số không được áp dụng
        # weight decay (L2 regularization)
        self.AT_log._no_weight_decay = True

        self.AL_log = nn.Parameter(torch.log(AL)) # (ED, N)
        self.AL_log._no_weight_decay = True

        # D (direct term) skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # Sepecific Parameter for backward scan
        # Forward dùng AT/AL/D chung
        if self.double_scans:
            ATb = torch.arrange(1, self.d_state + 1,
                                dtype=torch.float32).repeat(self.d_inner, 1)
            ALb = ATb.detach().clone()

            self.ATb_log_b = nn.Parameter(torch.log(ATb))
            self.ATb_log_b._no_weight_decay = True

            self.ALb_log_b = nn.Parameter(torch.log(ALb))
            self.ALb_log_b._no_weight_decay = True

            self.D_b = nn.Parameter(torch.ones(self.d_inner))
            self.D_b._no_weight_decay = True

            self.x_proj_b = nn.Linear(self.d_inner, 2*self.dt_rank + 2*self.d_state)

            self.dt_projT_b = nn.Linear(self.dt_rank, self.dt_inner)

            self.dt_projL_b = nn.Linear(self.dt_rank, self.dt_inner)
            if dt_init == "constant":
                nn.init.constant_(self.dt_projT_b.weight, dt_init_std)
                nn.init.constant_(self.dt_projL_b.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(self.dt_projT_b.weight, -dt_init_std, dt_init_std)
                nn.init.uniform_(self.dt_projL_b.weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError

            with torch.no_grad():# Turn off gradient tracking in this block 
                self.dt_projT_b.bias.copy_(inv_dt)
                self.dt_projL_b.bias.copy_(inv_dt)

    def proj_and_discretise(self, x: torch.Tensor):
        # Param: x = (B, H, W, ED) ED==d_inner
        # return: 
        # deltaAT, deltaAL, BXT, BXL, C
        # shapes: (B, H, W, ED, N) / (B, H, W, ED, N) / ... / (B, H, W, N)
        
        AT = -torch.exp(self.AT_log.float()) # (ED, N)
        AL = -torch.exp(self.AL_log.float())

        # (B, H, W, 2*dt_rank + 2*d_state)
        delta2BC = self.x_proj(x)

        deltaT, dettaL, B, C = torch.split(
                delta2BC,
                [self.dt_rank, self.dt_rank, self.dt_state, self.dt_state],
                dim=-1,
                )
        # (B, H, W, ED)
        deltaT = F.softplus(self.dt_projT(deltaT))
        deltaL = F.softplus(self.dt_projL(deltaL))

        # (B, H, W, ED, N)
        # unsqueeze(-1) chỉ thêm chiều để broadcasting
        deltaAT = torch.exp(deltaT.unsqueeze(-1) * AT)
        deltaAL = torch.exp(deltaL.unsqueeze(-1) * AL)

        # B: (B,H,W,N) -> (B,H,W,1,N) sau đó, broadcast theo ED
        deltaBT = deltaT.unsqueeze(-1)*B.unsqueeze(-2) # (B,H,W,ED,N)
        
        deltaBL = deltaL.unsqueeze(-1)*B.unsqueeze(-2) # (B,H,W,ED,N)

        BXT = deltaBT * x.unsqueeze(-1) # (B,H,W,ED,N)
        BXL = deltaBL * x.unsqueeze(-1) 

        # ép kiểu tránh mismatch
        deltaAT = deltaAT.type(dtype=x.dtype)
        deltaAL = deltaAL.type(dtype=x.dtype)
        BXT = BXT.type(dtype=x.dtype)
        BXL = BXL.type(dtype=x.dtype)
        
        return deltaAT, detaAL, BXT, BXL, C

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward scan #
        deltaAT, detaAL, BXT, BXL, C = self.proj_and_discretise(x)

        hs = wavefront_scan_cuda(deltaAT, deltaAL, BXT, BXL)

        y = (hs @ C.unsqueeze(-1)).unsqueeze(-1)
        
        # unsqueeze(0) để PyTorch biết broadcast như nào
        # D -> (1, 1, d_inner) | (1, 1, 1, d_inner)
        # (B, H, W, d_inner)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        
        # optioanl backward scan (bi-directional)
        if self.double_scans:
            x_b = torch.flip(x, dim=[1,2])
            deltaAT_b = deltaAL_b, BXT_b, BXL_b, C_b = self.proj_and_discretise(x_b)
            hs_b = wavefront_scan_cuda(
                    deltaAT_b, deltaAL_b, BXT_b, BXT_b
                    )
            y_b = (hs_b @ C_b.unsqueeze(-1)).squeeze(-1)
            y_b = y_b + x_b * self.D.unsqueeze(0).unsqueeze(0)

            # Avarage 2 direction
            y = ( y + y_b )/2
        return y
class M2D(nn.Module):
    '''
        Minimal version of Mamba 2D block, adapted from base 1D implementation:
        https://github.com/alxndrTL/mamba.py
    '''

    def __init__(self,
                 d_model: int,  # D | Số channel gốc của feature map
                 d_state: int = 16,  # N in paper/comments
                 expand_factor: int = 2,  # E in paper/
                 dt_rank: int | str = 'auto',
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 dt_init: str = "random",  # "random" or "constant"
                 dt_scale: float = 1.0,
                 dt_init_floor=1e-4,
                 double_scans: bool = False,  # Enable/Disable 2 scans
                 local_path: bool = False  # Enable/Disable local path
                 ):
        super().__init__()

        self.double_scans = double_scans
        self.d_model = d_model
        self.expand_factor = expand_factor
        self.d_inner = self.expand_factor * self.d_model  # E*D = ED in comments

        # Add SepConv local path
        self.local_norm = nn.LayerNorm(self.d_model) if local_path else None
        self.local_path = SepConv(self.d_model, expansion_ratio=2,
                                  act1_layer=StarReLU, act2_layer=StarReLU,
                                  bias=False, kernel_size=3, padding=1, residual=False) if local_path else None

        self.ssm_score = Mamba2DBlocK(
                d_inner         = self.d_inner,
                d_state         = d_state,
                dt_rank         = dt_rank,
                dt_min          = dt_minm
                dt_max          = dt_max,
                dt_init         = dt_init,
                dt_scale        = dt_scale,
                dt_init_floor   = dt_init_floor,
                double_scans    = double_scans,
                )
        self.in_block   = nn.Sequential(
                    nn.Linear(self.d_model, self.d_inner)
                    nn.GELU()
                )
        self.out_block  = nn.Sequential(
                    nn.GELU()
                    nn.Linear(self.d_inner, self.d_model)
                )
        """
        self.in_proj    = nn.Linear(self.d_model, self.d_inner) 
        self.act1       = nn.GELU()
        self.act2       = nn.GELU()
        self.out_proj   = nn.Linear(self.d_inner, self.d_model)
        """
    def forward(self, x):
        if self.local_path:
            local_features = self.local_path(self.local_norm(x))
        '''
        x = self.in_proj(x)
        x = self.act1(x)
        x = self.ssm_score(x)
        x = self.act2(x)
        x = self.out_proj(x)
        '''
        x = self.in_block(x)
        x = self.ssm_score(x)
        x = self.out_block(x)
        if self.local_path:
            x = x + local_features
        return x
