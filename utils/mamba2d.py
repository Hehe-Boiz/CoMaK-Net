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

class M2DBlock(nn.Module):
    '''
        Minimal version of Mamba 2D block, adapted from base 1D implementation:
        https://github.com/alxndrTL/mamba.py
    '''

    def __init__(self,
                 d_model: int,  # D
                 d_state: int = 16,  # N in paper/comments
                 expand_factor: int = 2,  # E in paper/comments
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
        self.d_state = d_state
        self.expand_factor = expand_factor
        self.d_inner = self.expand_factor * self.d_model  # E*D = ED in comments
        self.dt_rank = dt_rank

        # Add SepConv local path
        self.local_norm = nn.LayerNorm(self.d_model) if local_path else None
        self.local_path = SepConv(self.d_model, expansion_ratio=2,
                                  act1_layer=StarReLU, act2_layer=StarReLU,
                                  bias=False, kernel_size=3, padding=1, residual=False) if local_path else None

        # projects block input from D to ED (one_branch)
        self.in_proj = nn.Linear(self.d_model, self.d_inner)

        self.act1 = nn.GELU()

        # delta bias
        if self.dt_rank == 'auto': self.dt_rank = math.ceil(self.d_model / 16)

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))

        dt_init_std = self.dt_rank ** -0.5 * dt_scale

        # projects x to input-dependent deltaT, deltaL, B, C
        self.x_proj = nn.Linear(self.d_inner, 2 * self.dt_rank + 2 * self.d_state)

        # projects deltaT from dt_rank to d_inner
        self.dt_projT = nn.Linear(self.dt_rank, self.d_inner)

        # projects deltaL from dt_rank to d_inner
        self.dt_projL = nn.Linear(self.dt_rank, self.d_inner)

        # Init dt_projs
        if dt_init == "constant":
            nn.init.constant_(self.dt_projT.weight, dt_init_std)
            nn.init.constant_(self.dt_projL.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_projT.weight, -dt_init_std, dt_init_std)
            nn.init.uniform_(self.dt_projL.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        with torch.no_grad():
            self.dt_projT.bias.copy_(inv_dt)
            self.dt_projL.bias.copy_(inv_dt)

        # S4D real initialization
        AT = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        AL = AT.detach().clone()

        self.AT_log = nn.Parameter(torch.log(AT))
        self.AT_log._no_weight_decay = True

        self.AL_log = nn.Parameter(torch.log(AL))
        self.AL_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # Init second set of params for bwd scan direction
        if self.double_scans:
            # S4D real initialization
            ATb = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
            ALb = ATb.detach().clone()

            self.AT_log_b = nn.Parameter(torch.log(ATb))
            self.AT_log_b._no_weight_decay = True

            self.AL_log_b = nn.Parameter(torch.log(ALb))
            self.AL_log_b._no_weight_decay = True

            self.D_b = nn.Parameter(torch.ones(self.d_inner))
            self.D_b._no_weight_decay = True

            # projects x to input-dependent deltaT, deltaL, B, C
            self.x_proj_b = nn.Linear(self.d_inner, 2 * self.dt_rank + 2 * self.d_state)

            # projects deltaT from dt_rank to d_inner
            self.dt_projT_b = nn.Linear(self.dt_rank, self.d_inner)

            # projects deltaL from dt_rank to d_inner
            self.dt_projL_b = nn.Linear(self.dt_rank, self.d_inner)

            # Init dt_projs
            if dt_init == "constant":
                nn.init.constant_(self.dt_projT_b.weight, dt_init_std)
                nn.init.constant_(self.dt_projL_b.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(self.dt_projT_b.weight, -dt_init_std, dt_init_std)
                nn.init.uniform_(self.dt_projL_b.weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError

            with torch.no_grad():
                self.dt_projT_b.bias.copy_(inv_dt)
                self.dt_projL_b.bias.copy_(inv_dt)

        self.act2 = nn.GELU()

        # projects block output from ED back to D
        self.out_proj = nn.Linear(self.d_inner, self.d_model)

    def forward(self, x):
        # x : (B, H, W, ED)

        if self.local_path:
            local_features = self.local_path(self.local_norm(x))

        x = self.in_proj(x)
        x = self.act1(x)
        x = self.ssm(x)
        x = self.act2(x)
        x = self.out_proj(x)

        if self.local_path:
            x = x + local_features

        return x

    def proj_and_discretise(self, x):
        # x : (B, H, W, ED)

        AT = -torch.exp(self.AT_log.float())  # (ED, N)
        AL = -torch.exp(self.AL_log.float())  # (ED, N)

        delta2BC = self.x_proj(x)  # (B, H, W, dt_rank+2*2N)
        # Splits proj: (B, H, W, 2*dt_rank+N) -> 2*(B, H, W, dt_rank), 2*(B, H, W, N)
        deltaT, deltaL, B, C = torch.split(delta2BC, [self.dt_rank,
                                                      self.dt_rank,
                                                      self.d_state,
                                                      self.d_state],
                                           dim=-1)

        deltaT = F.softplus(self.dt_projT(deltaT))
        deltaL = F.softplus(self.dt_projL(deltaL))

        deltaAT = torch.exp(deltaT.unsqueeze(-1) * AT)  # (B, H, W, ED, N)
        deltaAL = torch.exp(deltaL.unsqueeze(-1) * AL)  # (B, H, W, ED, N)

        deltaBT = deltaT.unsqueeze(-1) * B.unsqueeze(-2)  # (B, H, W, ED, N)
        deltaBL = deltaL.unsqueeze(-1) * B.unsqueeze(-2)  # (B, H, W, ED, N)

        BXT = deltaBT * x.unsqueeze(-1)  # (B, H, W, ED, N)
        BXL = deltaBL * x.unsqueeze(-1)  # (B, H, W, ED, N)

        # Force casting of types for mixed pres. training
        deltaAT = deltaAT.type(dtype=x.dtype)
        deltaAL = deltaAL.type(dtype=x.dtype)
        BXT = BXT.type(dtype=x.dtype)
        BXL = BXL.type(dtype=x.dtype)

        return (deltaAT, deltaAL, BXT, BXL, C)

    def ssm(self, x):
        # x : (B, H, W, ED)
        # y : (B, H, W, ED)

        deltaAT, deltaAL, BXT, BXL, C = self.proj_and_discretise(x)

        hs = wavefront_scan_cuda(deltaAT, deltaAL, BXT, BXL)  # (B, H, W, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(-1)

        y = y + x * self.D.unsqueeze(0).unsqueeze(0)

        if self.double_scans:
            # Implement bwd scan as flipped input
            x_b = torch.flip(x, dims=[1, 2])

            deltaAT_b, deltaAL_b, BXT_b, BXL_b, C_b = self.proj_and_discretise(x_b)

            hs_b = wavefront_scan_cuda(deltaAT_b, deltaAL_b, BXT_b, BXL_b)  # (B, H, W, ED, N)

            y_b = (hs_b @ C_b.unsqueeze(-1)).squeeze(-1)

            y_b = y_b + x_b * self.D.unsqueeze(0).unsqueeze(0)

            y = (y + y_b) / 2  # average output of both scans

        return y