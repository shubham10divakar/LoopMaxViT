"""
blocks.py: MaxViT building blocks with M1 PMAL (PyTorch port of the design's `common_blocks.py` / `loop_maxvit.py`).

Layouts: MBConv works on (B,C,H,W) maps; the two attention axes work channels-last on (B,H,W,C).
Every loopable block follows the block protocol in looped_core.py: block(x, t=<int>, state=<obj|None>) -> y or
(y, state). The MaxViT block takes and returns (B,C,H,W) maps.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import pmal
from .looped_core import IterNorm


class Mlp(nn.Sequential):
    def __init__(self, dim, ratio=4.0):
        super().__init__(nn.Linear(dim, int(dim * ratio)), nn.GELU(), nn.Linear(int(dim * ratio), dim))


class SE(nn.Module):
    def __init__(self, ch, hidden):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, ch, 1)

    def forward(self, x):
        s = x.mean((2, 3), keepdim=True)
        return x * torch.sigmoid(self.fc2(F.silu(self.fc1(s))))


# -------------------------------------------------------------- MBConv (loopable)
class MBConv(nn.Module):
    """Pre-norm MBConv (MaxViT style: expand 4, SE 0.25) with per-iteration BN (R4).
    The stride-2 version is only used in the unique first block of a stage (never looped)."""

    def __init__(self, c_in, c_out, t_max=1, expand=4, se=0.25, stride=1):
        super().__init__()
        mid = c_in * expand
        self.n0 = IterNorm(c_in, t_max)
        self.pw1 = nn.Conv2d(c_in, mid, 1, bias=False)
        self.n1 = IterNorm(mid, t_max)
        self.dw = nn.Conv2d(mid, mid, 3, stride, 1, groups=mid, bias=False)
        self.n2 = IterNorm(mid, t_max)
        self.se = SE(mid, max(8, int(c_in * se))) if se else None
        self.pw2 = nn.Conv2d(mid, c_out, 1)
        self.short = None
        if stride > 1 or c_in != c_out:
            pool = nn.AvgPool2d(stride, ceil_mode=True) if stride > 1 else nn.Identity()
            self.short = nn.Sequential(pool, nn.Conv2d(c_in, c_out, 1))

    def forward(self, x, t=0, state=None):
        h = self.n0(x, t=t)
        h = F.gelu(self.n1(self.pw1(h), t=t))
        h = F.gelu(self.n2(self.dw(h), t=t))
        if self.se is not None:
            h = self.se(h)
        h = self.pw2(h)
        return (x if self.short is None else self.short(x)) + h


# ------------------------------------------------------------ partition helpers (B,H,W,C)
def window_partition(x, p):
    """(B,H,W,C) -> (B*H/p*W/p, p*p, C): contiguous p x p windows (local, dense)."""
    B, H, W, C = x.shape
    x = x.view(B, H // p, p, W // p, p, C).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, p * p, C)


def window_reverse(x, p, h, w):
    C = x.shape[-1]
    x = x.view(-1, h // p, w // p, p, p, C).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, h, w, C)


def grid_partition(x, g):
    """(B,H,W,C) -> (B*H/g*W/g, g*g, C): g x g dilated grid (global, sparse); groups share (i mod H/g, j mod W/g)."""
    B, H, W, C = x.shape
    x = x.view(B, g, H // g, g, W // g, C).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(-1, g * g, C)


def grid_reverse(x, g, h, w):
    C = x.shape[-1]
    x = x.view(-1, h // g, w // g, g, g, C).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(-1, h, w, C)


PARTITION = {"window": (window_partition, window_reverse), "grid": (grid_partition, grid_reverse)}


def spatial_permute(x, perm):
    """Gather the H*W positions of a (B,H,W,C) map: position k of the output holds input position perm[k]."""
    B, H, W, C = x.shape
    return x.reshape(B, H * W, C).index_select(1, perm).view(B, H, W, C)


# ------------------------------------------- MHSA with 2D relative position bias
def rel_index(h, w):
    """Index into a (2h-1)(2w-1) relative-offset table for every (query, key) pair."""
    coords = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing="ij")).reshape(2, -1)
    rel = coords[:, :, None] - coords[:, None, :]
    rel[0] += h - 1
    rel[1] += w - 1
    return torch.from_numpy((rel[0] * (2 * w - 1) + rel[1]).reshape(-1).astype("int64"))


class RelPosMHSA(nn.Module):
    """softmax(QK^T/sqrt(d) + B_rel[table] + prior + mask) V over a fixed (h, w) token group.

    n_tables > 1 gives each loop iteration its own relative-bias table (sec. 4.3: PMAL changes the true
    offsets inside a group, so iteration t learns its own bias). prior: loop-carried logits (M3).
    mask: additive (nG, N, N) mask repeated over the batch (PM3 Swin-style masked windows).
    Returns (out, logits) when return_logits=True. Without prior/logits it uses fused SDPA.
    If `self.record` is a list, the attention probabilities are appended to it (attention rollout, PM5).
    """

    def __init__(self, dim, heads, size, n_tables=1):
        super().__init__()
        self.dim, self.heads, self.dh = dim, heads, dim // heads
        self.h, self.w = size
        self.n = self.h * self.w
        self.n_tables = n_tables
        self.record = None
        self.register_buffer("idx", rel_index(self.h, self.w), persistent=False)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.rel_table = nn.Parameter(torch.zeros(n_tables, (2 * self.h - 1) * (2 * self.w - 1), heads))
        nn.init.trunc_normal_(self.rel_table, std=0.02)

    def bias(self, table_id=0):
        b = self.rel_table[min(table_id, self.n_tables - 1)][self.idx]          # (N*N, heads)
        return b.view(self.n, self.n, self.heads).permute(2, 0, 1)             # (heads, N, N)

    def forward(self, x, prior=None, mask=None, table_id=0, return_logits=False):
        B, N, _ = x.shape
        assert N == self.n, f"RelPosMHSA built for {self.h}x{self.w} tokens, got N={N}"
        q, k, v = self.qkv(x).view(B, N, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        bias = self.bias(table_id).to(q.dtype).unsqueeze(0)                    # (1, heads, N, N)
        if mask is not None:
            bias = bias + mask.to(q.dtype).unsqueeze(1).repeat(B // mask.shape[0], 1, 1, 1)
        if prior is None and not return_logits and self.record is None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
            logits = None
        else:
            logits = (q @ k.transpose(-2, -1)) * self.dh ** -0.5 + bias
            if prior is not None:
                logits = logits + prior.to(logits.dtype)
            attn = logits.float().softmax(-1)
            if self.record is not None:
                self.record.append(attn.detach())
            out = attn.to(v.dtype) @ v
        out = self.proj(out.transpose(1, 2).reshape(B, N, self.dim))
        return (out, logits) if return_logits else out


# ------------------------------------------------------------ one attention axis (+ FFN)
class AxisAttention(nn.Module):
    """x <- x + pi_t^-1( Attn( partition( pi_t( LN x ) ) ) ),  x <- x + FFN(LN x)     on (B,H,W,C).

    kind      "window" (block attention, P x P) or "grid" (grid attention, G x G)
    permuted  M1 PMAL on this axis: iteration t > 0 uses the precomputed permutation pi_t (t = 0 is canonical)
    n_tables  relative-bias tables (t_max with PMAL, 1 otherwise; ablation PM4)
    carry     M3: L_t = ... + alpha_t[h] * stopgrad(L_{t-1}); alpha init 0. State = this axis' logits.
    masked    PM3: Swin mask instead of cyclic windows (window axis with PMAL only)
    """

    def __init__(self, dim, heads, map_size, kind, size, t_max=1, permuted=False, n_tables=1, carry=False,
                 masked=False, detach=True):
        super().__init__()
        self.kind, self.size, self.permuted, self.carry, self.detach = kind, size, permuted, carry, detach
        self.h, self.w = map_size
        assert self.h % size == 0 and self.w % size == 0, \
            f"{kind} size {size} must divide the {self.h}x{self.w} map (see doc 02 sec. 5 compatibility table)"
        self.part, self.rev = PARTITION[kind]
        self.ln, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.att = RelPosMHSA(dim, heads, (size, size), n_tables=n_tables)
        self.mlp = Mlp(dim)
        self.alpha = nn.Parameter(torch.zeros(t_max, heads)) if carry else None
        self.masked = masked and permuted and kind == "window"
        if permuted:
            perm = pmal.schedule(kind, t_max, self.h, self.w, size)
            self.register_buffer("perm", torch.from_numpy(perm), persistent=False)
            self.register_buffer("inv", torch.from_numpy(np.stack([pmal.inverse_perm(p) for p in perm])),
                                 persistent=False)
        if self.masked:
            m = [pmal.window_mask(self.h, self.w, size, pmal.window_shift(t, size)) for t in range(t_max)]
            self.register_buffer("mask", torch.from_numpy(np.stack(m)), persistent=False)

    def perm_index(self, t):
        """Gather index used at iteration t (None = canonical layout)."""
        return self.perm[min(t, len(self.perm) - 1)] if self.permuted and t > 0 else None

    def group_index(self, t):
        """(nG, group size) ORIGINAL row-major position of every attention slot at iteration t (for rollout)."""
        pos = torch.arange(self.h * self.w, device=self.ln.weight.device).view(1, self.h, self.w, 1).float()
        p = self.perm_index(t)
        if p is not None:
            pos = spatial_permute(pos, p)
        return self.part(pos, self.size)[..., 0].long()

    def forward(self, x, t=0, state=None):
        h = self.ln(x)
        p = self.perm_index(t)
        if p is not None:
            h = spatial_permute(h, p)
        prior = None
        if self.carry and state is not None:
            prev = state.detach() if self.detach else state
            prior = self.alpha[min(t, len(self.alpha) - 1)].view(1, -1, 1, 1) * prev
        mask = self.mask[min(t, len(self.mask) - 1)] if self.masked and t > 0 else None
        out = self.att(self.part(h, self.size), prior=prior, mask=mask, table_id=t, return_logits=self.carry)
        out, logits = out if self.carry else (out, None)
        h = self.rev(out, self.size, self.h, self.w)
        if p is not None:
            h = spatial_permute(h, self.inv[min(t, len(self.inv) - 1)])
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x, logits


# ------------------------------------------------------------------ the MaxViT block (loop unit)
class MaxViTBlock(nn.Module):
    """MBConv -> block attention (P x P windows) + FFN -> grid attention (G x G grid) + FFN   (doc 02 sec. 2).

    The whole block is the loop unit (macro-block, R3b): conv and attention alternate every iteration.
    pmal_axes: axes that get M1 permutations, e.g. ("window", "grid") (Loop-full), () (Loop-plain),
               ("window",) / ("grid",) for ablation PM2.
    rel_tables: "auto" (per-iteration tables on permuted axes), "shared" or "per_iter" (ablation PM4).
    carry: M3 per-axis loop-carried logits; state = (window logits, grid logits).
    (B,C_in,H_in,W_in) map in, (B,dim,H,W) map out; size = (H, W) after the optional stride-2 MBConv.
    """

    def __init__(self, c_in, dim, size, P=7, G=7, t_max=1, stride=1, pmal_axes=(), rel_tables="auto",
                 carry=False, masked=False, head_dim=32):
        super().__init__()
        heads = max(1, dim // head_dim)
        self.carry = carry
        self.mb = MBConv(c_in, dim, t_max=t_max, stride=stride)

        def axis(kind, s):
            perm = kind in pmal_axes
            n_tab = {"auto": t_max if perm else 1, "shared": 1, "per_iter": t_max}[rel_tables]
            return AxisAttention(dim, heads, size, kind, s, t_max, perm, n_tab, carry, masked)

        self.block_att = axis("window", P)
        self.grid_att = axis("grid", G)

    def forward(self, x, t=0, state=None):
        x = self.mb(x, t=t)
        sb, sg = state if state is not None else (None, None)
        h = x.permute(0, 2, 3, 1)
        h, lb = self.block_att(h, t=t, state=sb)
        h, lg = self.grid_att(h, t=t, state=sg)
        y = h.permute(0, 3, 1, 2).contiguous()
        return (y, (lb, lg)) if self.carry else y
