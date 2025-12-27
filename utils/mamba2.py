import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from utils.utils import RoPE, Scale, to_ttensor

# Increase torch.compile cache size limit
torch._dynamo.config.cache_size_limit = 64


class Mamba2(nn.Module):
    def __init__(
        self,
        d_model,
        ngroups=1,
        A_init_range=(1, 16),
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        bias=False,
        conv_bias=True,
        d_state=64,
        dt_min=0.001,
        dt_max=0.1,
        activation="silu",  # default to silu
        dt_init_floor=1e-4,
        expand=2,  ## d_inner
        headdim=64,  # default to 64
        d_conv=3,  # default to 3 for 2D
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=False,  # default to False, for custom implementation
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
        **kwargs,
    ):
        ##
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.headdim = headdim
        self.d_state = d_state
        if ngroups == -1:
            ngroups = self.d_inner // self.headdim  # equivalent to multi-head attention
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        # convert chunk_size to triton.language.int32
        self.chunk_size = chunk_size  # torch.tensor(chunk_size,dtype=torch.int32)
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        self.partial_win_size = kwargs.get("partial_win_size", -1)  # default to -1
        self.win_only = kwargs.get("win_only", False)  # default to False
        self.ssd_aexp = kwargs.get("ssd_aexp", False)  # default to 2
        self.ssd_positve_dA = kwargs.get(
            "ssd_positve_dA", False
        )  # default to False, ablation for linear attn duality
        self.ssd_norm_da = kwargs.get("ssd_norm_da", False)
        self.ssd_linear_norm = kwargs.get("ssd_linear_norm", False)
        self.win_norm = kwargs.get("win_norm", False)
        self.zact = kwargs.get("zact", False)
        if self.ssd_linear_norm:
            self.elu = nn.ELU()
        else:
            # Order: [x, B, C, dt]
            d_in_proj = 1 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(
            self.d_model, int(d_in_proj), bias=bias, **factory_kwargs
        )  #
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state

        if self.learnable_init_states:
            self.init_states = nn.Parameter(
                torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs)
            )
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(
            *A_init_range
        )
        # A_log (nheads) or (d_inner, d_state)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        if kwargs.get("dscale", False):
            self.D = Scale(dim=self.d_inner, init_value=1.0, trainable=True)
        else:
            self.D = nn.Parameter(torch.ones(self.nheads, device=device))
            self.D._no_weight_decay = True
        # modified from RMSNormGated to layer norm
        # assert RMSNormGated is not None
        # self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)
        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )

        # linear attention duality
        # self.linear_attn_duality = linear_attn_duality

        # lepe positional encoding
        if kwargs.get("lepe", False):
            self.lepe = nn.Conv2d(
                self.d_inner, self.d_inner, 3, padding=1, groups=self.d_inner
            )
        else:
            self.lepe = None
        if kwargs.get("rope", False):
            HW: tuple[int, int] = cast(
                tuple[int, int], kwargs.get("input_resolution")
            )  # FIXME: fix the resolution in dynamic input
            self.ropes = RoPE(shape=(HW[0], HW[1], self.d_state), base=10000)
        else:
            self.ropes = None
        self.ab_bias = kwargs.get("ab_bias", False)
        self.decouple_hw = kwargs.get("decouple_hw", False)
        self.kwargs = kwargs

    def non_casual_linear_attn(self, x, dt, A, B, C, D, H=None, W=None):
        """
        non-casual attention duality of mamba v2
        x: (B, L, H, D), equivalent to V in attention
        dt: (B, L, nheads)
        A: (nheads) or (d_inner, d_state)
        B: (B, L, d_state), equivalent to K in attention
        C: (B, L, d_state), equivalent to Q in attention
        D: (nheads), equivalent to the skip connection
        """
        skip = x
        batch, seqlen, head, dim = x.shape
        dstate = B.shape[2]
        V = x.permute(0, 2, 1, 3)  # (B, H, L, D)
        dt = dt.permute(0, 2, 1)  # (B, H, L)
        # dA = exp(dt * A)
        dA = dt.unsqueeze(-1) * A.view(
            1, -1, 1, 1
        )  # .repeat(batch, 1, seqlen, 1) # (B, H, L, 1)
        if self.ssd_aexp:
            dA = 1 / dA.exp()
        if self.ssd_positve_dA:
            dA = -dA
        if self.ssd_norm_da:
            dA = dA / torch.sum(dA, dim=-2, keepdim=True)

        K = B.view(batch, 1, seqlen, dstate)  # (B, 1, L, D)

        Q = C.view(batch, 1, seqlen, dstate)  # (B, 1, L, dstate)
        # SSD core logic
        if self.ropes is not None:
            Q = (
                self.ropes(Q.view(batch, H, W, dstate))
                .view(batch, -1, dstate)
                .unsqueeze(1)
            )
            K = (
                self.ropes(K.view(batch, H, W, dstate))
                .view(batch, -1, dstate)
                .unsqueeze(1)
            )

        if self.kwargs.get("exp_da", False):
            dA = dA.softmax(dim=-2)
            Kscaled = K * dA
            KV = Kscaled.transpose(-2, -1) @ V
            x = Q @ KV
        else:
            V_scaled = V * dA

            if Q.dtype != V_scaled.dtype or Q.dtype != V_scaled.dtype:
                Q, K = Q.to(V_scaled.dtype), K.to(V_scaled.dtype)
            KV = K.transpose(-2, -1) @ V_scaled  # (B, H, dstate, D)
            x = Q @ KV  # (B, H, L, D)
        if self.kwargs.get("dscale", False):
            x = x.permute(0, 2, 1, 3).contiguous() + self.D(skip.flatten(2, 3)).view(
                batch, seqlen, head, dim
            )
        else:
            x = x + V * D.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)
            x = x.permute(0, 2, 1, 3).contiguous()  # (B, L, H, D)
        return x, KV

    def forward(self, x: torch.Tensor):
        """
        x: (B,H,W,D) _ dimension
        Returns: same shape as x
        """
        B, H, W, D = x.shape
        u = x.view(B, H * W, D)
        A = -torch.exp(self.A_log)
        # Linear(d_model -> d_inner)
        xbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        xBC, dt = torch.split(
            xbcdt, [self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
        )
        dt = F.softplus(dt + self.dt_bias)
        assert self.activation in ["silu", "swish"]

        x_ssm, B_ssm, C_ssm = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )
        x_ssm, dt, A, B_ssm, C_ssm = to_ttensor(x_ssm, dt, A, B_ssm, C_ssm)
        y, _ = self.non_casual_linear_attn(
            rearrange(x_ssm, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A,
            B_ssm,
            C_ssm,
            self.D,
            H,
            W,
        )
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y)
        out = self.out_proj(y)  # (B, L, D)
        out = out.view(B, H, W, D)
        return out
