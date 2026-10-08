"""
looped_core.py: shared looping machinery (identical to the LoopCoAtNet repo except tuple states) (PyTorch port of the design's Keras `looped_core.py`, doc 00 sec. 6).

  * IterNorm      per-iteration BatchNorm (rule R4)
  * LoopedStage   ONE shared block applied T times inside a resolution stage, with
                  iteration embedding e_t, input injection beta_t, per-iteration step gamma_t,
                  optional loop-carried state, truncated BPTT (R7), per-iteration stochastic depth (R8)
                  and a residual trace (fixed-point diagnostic)
  * UnrolledStage the baseline counterpart: N distinct blocks, same call signature
  * CycleBlock    k x T granularity (R3c)
  * sample_T / recalibrate_bn   random-T training (R6) and exact BN recalibration (R13)

Block protocol (every block that lives inside a stage):
    y          = block(x, t=t, state=None)
    (y, state) = block(x, t=t, state=state)     # blocks that carry state (M3; may be a tuple)
  x and y have the same shape, (B,C,H,W) maps or (B,N,C) tokens. The block is residual internally.
  t only selects cheap per-iteration parameters; heavy weights are shared.

Guarantee (tests/test_model.py): with fresh init (e=0, beta=0, gamma=1) and T=1,
LoopedStage(block)(x) == block(x) exactly.
"""
import random

import torch
import torch.nn as nn


def drop_path(x, rate, training):
    """Stochastic depth on a residual update, per sample."""
    if not training or rate <= 0.0:
        return x
    keep = 1.0 - rate
    mask = x.new_empty((x.shape[0],) + (1,) * (x.dim() - 1)).bernoulli_(keep)
    return x * mask / keep


# ----------------------------------------------------------------------------- norms
class IterNorm(nn.Module):
    """t_max separate BatchNorms (Recurrent-BN, Cooijmans et al. 2016).

    Activation statistics differ per iteration, so a shared BN's running stats are wrong at
    test time. Un-sharing costs 2C params per iteration. For t >= t_max the last BN is reused.
    Unlike the Keras version, all BNs are created eagerly, so R10 (build at t_max) is not needed.
    """

    def __init__(self, ch, t_max=1, eps=1e-5, momentum=0.1):
        super().__init__()
        self.norms = nn.ModuleList([nn.BatchNorm2d(ch, eps=eps, momentum=momentum) for _ in range(t_max)])

    def forward(self, x, t=0):
        return self.norms[min(t, len(self.norms) - 1)](x)


# ----------------------------------------------------------------------- looped stage
def _per_channel(p, x):
    """Broadcast a (C,) parameter against (B,C,H,W) maps or (B,N,C) tokens."""
    return p.view(1, -1, 1, 1) if x.dim() == 4 else p


def _detach(state):
    """Detach a loop-carried state: a tensor, a tuple of tensors (per-axis logits) or None."""
    if isinstance(state, (tuple, list)):
        return type(state)(_detach(s) for s in state)
    return state.detach() if state is not None else None


class LoopedStage(nn.Module):
    """x_{t+1} = x_t + gamma_t * ( B(x~_t; t) - x~_t ),   x~_t = x_t + e_t + beta_t * x_0

    e_t     iteration embedding (C,)    init 0
    beta_t  input-injection gate (C,)   init 0   (re-anchors to x_0, prevents drift at large T)
    gamma_t per-iteration step (C,)     init 1   (LayerScale per iteration)
    """

    def __init__(self, block, dim, t_train, t_max=None, trunc_k=None, drop_path=(0.0, 0.0)):
        super().__init__()
        self.block = block
        self.t_train = int(t_train)
        self.t_max = int(t_max or 2 * t_train)   # R9: parameters allocated for extrapolation
        self.trunc_k = trunc_k                     # R7: backprop only through the last k iterations
        self.dp_lo, self.dp_hi = drop_path         # R8: rate rises linearly with t
        self.iter_emb = nn.Parameter(torch.zeros(self.t_max, dim))
        self.beta = nn.Parameter(torch.zeros(self.t_max, dim))
        self.gamma = nn.Parameter(torch.ones(self.t_max, dim))

    def forward(self, x0, T=None, return_trace=False):
        T = int(T or self.t_train)
        x, state, trace = x0, None, []
        for t in range(T):
            ti = min(t, self.t_max - 1)
            xt = x + _per_channel(self.iter_emb[ti], x) + _per_channel(self.beta[ti], x) * x0
            out = self.block(xt, t=ti, state=state)
            y, state = out if isinstance(out, tuple) else (out, None)
            rate = self.dp_lo + (self.dp_hi - self.dp_lo) * t / max(1, T - 1)
            x_new = x + drop_path(_per_channel(self.gamma[ti], x) * (y - xt), rate, self.training)
            if return_trace:  # relative update size; should shrink with t (fixed-point check)
                trace.append((x_new - x).detach().float().norm() / (x.detach().float().norm() + 1e-6))
            x = x_new
            if self.trunc_k and t < T - self.trunc_k:
                x = x.detach()
                state = _detach(state)
        return (x, trace) if return_trace else x


class UnrolledStage(nn.Module):
    """Baseline counterpart of LoopedStage: N distinct blocks, same call signature (T is ignored).
    Used for the Base and Shallow-matched variants (doc 00, sec. 8.2)."""

    def __init__(self, blocks, drop_path=(0.0, 0.0)):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.dp_lo, self.dp_hi = drop_path

    def forward(self, x, T=None, return_trace=False):
        state, trace, n = None, [], len(self.blocks)
        for i, b in enumerate(self.blocks):
            out = b(x, t=0, state=state)
            y, state = out if isinstance(out, tuple) else (out, None)
            rate = self.dp_lo + (self.dp_hi - self.dp_lo) * i / max(1, n - 1)
            x_new = x + drop_path(y - x, rate, self.training)
            if return_trace:
                trace.append((x_new - x).detach().float().norm() / (x.detach().float().norm() + 1e-6))
            x = x_new
        return (x, trace) if return_trace else x


class CycleBlock(nn.Module):
    """k x T granularity (R3c): k distinct shared blocks applied cyclically A,B,A,B,...
    Iteration t uses blocks[t % k] with per-block iteration index t // k."""

    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, t=0, state=None):
        k = len(self.blocks)
        return self.blocks[t % k](x, t=t // k, state=state)


def make_stage(block_factory, n_blocks, loop, dim, t_train=None, t_max=None, trunc_k=None,
               drop_path=(0.0, 0.0)):
    """block_factory(t_max) -> block.
    loop=True  -> LoopedStage(1 shared block, T = t_train or n_blocks)
    loop=False -> UnrolledStage(n_blocks distinct blocks)."""
    if loop:
        t_train = t_train or n_blocks
        t_max = t_max or 2 * t_train
        return LoopedStage(block_factory(t_max), dim, t_train, t_max, trunc_k, drop_path)
    return UnrolledStage([block_factory(1) for _ in range(n_blocks)], drop_path)


# ---------------------------------------------------------------- random-T training
def sample_T(spec):
    """spec: {"stage3": (lo, hi), ...} -> {"stage3": t}, sampled once per batch (R6)."""
    return {k: random.randint(lo, hi) for k, (lo, hi) in spec.items()}


@torch.no_grad()
def recalibrate_bn(model, batches, T=None, device="cuda", amp_dtype=None):
    """Rule R13. Recompute every BatchNorm's running statistics as the exact cumulative average
    over `batches` (iterable of input tensors or (x, y) pairs), at the loop counts T used for
    evaluation. Only the BNs run in train mode; DropPath and everything else stay in eval mode.
    Per-iteration BNs that T does not reach are left reset (mean 0, var 1)."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    was_training = model.training
    old_momentum = [b.momentum for b in bns]
    model.eval()
    for b in bns:
        b.reset_running_stats()
        b.momentum = None                    # PyTorch: None = exact cumulative moving average
        b.train()
    n, dev_type = 0, torch.device(device).type
    for xb in batches:
        if isinstance(xb, (tuple, list)):
            xb = xb[0]
        with torch.autocast(device_type=dev_type, dtype=amp_dtype, enabled=amp_dtype is not None):
            model(xb.to(device, non_blocking=True), T=T)
        n += 1
    for b, mo in zip(bns, old_momentum):
        b.momentum = mo
    model.train(was_training)
    return n
