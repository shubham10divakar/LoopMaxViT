"""
model.py: LoopMaxViT (design doc 02), PyTorch.

MaxViT (Tu et al., ECCV 2022): conv stem (112^2) | S1..S4, each = stride-2 MaxViT block + (L-1) MaxViT blocks.
MaxViT block = MBConv -> block attention (P x P windows) + FFN -> grid attention (G x G grid) + FFN, head dim 32.
Looping (doc 02 sec. 3): every stage = unique stride-2 transition block + LoopedStage(1 shared MaxViT block)
(R1: transitions stay unique; R3b: the whole MaxViT block is the loop unit; R11: stem and head are never looped).

Modes (doc 00 sec. 8.2 model triplets, doc 02 sec. 5):
  base     Base MaxViT, distinct blocks everywhere
  shallow  Shallow-matched: 1 distinct block after each transition (matched params, fewer FLOPs)
  plain    Loop-plain: 1 shared MaxViT block x T over FIXED partitions
  pmal     Loop-full:  1 shared MaxViT block x T with M1 PMAL (window roll + grid block shear per iteration,
                       per-iteration relative-bias tables)
carry=True adds M3 per-axis loop-carried attention logits to the S3 block (plain / pmal).
Ablations (doc 02 sec. 7): pmal_axes = both | window | grid (PM2), window_mask (PM3, Swin-masked windows),
rel_tables = auto | shared | per_iter (PM4), s3_cycle = k (k x T granularity, R3c).
"""
import math

import torch
import torch.nn as nn

from .blocks import MaxViTBlock
from .looped_core import CycleBlock, LoopedStage, make_stage

CFG = {
    "maxvit-nano": dict(L=[2, 2, 5, 2], C=[32, 64, 128, 256], stem=32),     # Kaggle / dev size
    "maxvit-t": dict(L=[2, 2, 5, 2], C=[64, 128, 256, 512], stem=64),
    "maxvit-s": dict(L=[2, 2, 5, 2], C=[96, 192, 384, 768], stem=64),
    "maxvit-b": dict(L=[2, 6, 14, 2], C=[96, 192, 384, 768], stem=64),      # paper headline (-56 %)
}
MODES = ("base", "shallow", "plain", "pmal")
PMAL_AXES = {"both": ("window", "grid"), "window": ("window",), "grid": ("grid",)}
REL_TABLES = ("auto", "shared", "per_iter")
HEAD_DIM = 32
STAGES = ("stage1", "stage2", "stage3", "stage4")
# doc 02 sec. 3: S1 and S4 have one block after the transition; they loop with random T ~ U[1,2] / U[1,3]
SHORT_STAGE_RANDOM = {"stage1": (1, 2), "stage4": (1, 3)}


def _random_range(t_train):
    """Doc 00 sec. 7: T ~ U[ceil(0.5 T), ceil(1.5 T)] per batch (S3 of MaxViT-T: T = 4 -> U[2, 6])."""
    return max(1, math.ceil(0.5 * t_train)), max(1, math.ceil(1.5 * t_train))


def resolve_P(img, P="auto"):
    """Window / grid size. auto = img / 32 (224 -> 7, 192 -> 6, 160 -> 5, 128 -> 4, 64 -> 2; doc 02 sec. 5),
    which divides every stage map (img/4 .. img/32)."""
    P = img // 32 if P in (None, "auto") else int(P)
    for s in (4, 8, 16, 32):
        assert (img // s) % P == 0, f"P = G = {P} must divide every stage map ({img}px: {[img // d for d in (4, 8, 16, 32)]})"
    return P


class LoopMaxViT(nn.Module):
    """call: model(x, T={"stage1": .., "stage2": .., "stage3": .., "stage4": ..}); missing keys use t_train."""

    def __init__(self, variant="maxvit-t", img=224, num_classes=1000, mode="pmal", carry=False, T=None,
                 drop_path=0.0, s3_cycle=1, trunc_k="auto", P="auto", pmal_axes="both", rel_tables="auto",
                 window_mask=False, loop_stages=STAGES, in_chans=3):
        super().__init__()
        assert variant in CFG, f"variant must be one of {list(CFG)}"
        assert mode in MODES, f"mode must be one of {MODES}"
        assert pmal_axes in PMAL_AXES, f"pmal_axes must be one of {list(PMAL_AXES)}"
        assert rel_tables in REL_TABLES, f"rel_tables must be one of {REL_TABLES}"
        assert img % 32 == 0, "image size must be a multiple of 32 (S4 runs at img/32)"
        assert not (carry and mode in ("base", "shallow")), "carry (M3) needs a looped S3"
        assert s3_cycle == 1 or mode in ("plain", "pmal"), "s3_cycle > 1 needs a looped S3"
        assert not (window_mask and mode != "pmal"), "window_mask (PM3) only applies to PMAL-shifted windows"
        c = CFG[variant]
        L, C = c["L"], c["C"]
        self.variant, self.mode, self.carry, self.img, self.num_classes = variant, mode, carry, img, num_classes
        self.P = resolve_P(img, P)
        self.pmal_axes, self.rel_tables, self.window_mask, self.s3_cycle = pmal_axes, rel_tables, window_mask, s3_cycle
        loop = mode in ("plain", "pmal")
        axes = PMAL_AXES[pmal_axes] if mode == "pmal" else ()
        T = dict(T or {})

        # stochastic depth: linear over the baseline block positions (R8 inside loops)
        n_all = sum(L)
        pos = [0] + torch.tensor(L).cumsum(0).tolist()
        dp = lambda s: (drop_path * (pos[s] + 1) / max(1, n_all - 1), drop_path * (pos[s + 1] - 1) / max(1, n_all - 1))

        self.loop_spec = {}
        for i, s in enumerate(STAGES):
            n_after = 1 if mode == "shallow" else max(1, L[i] - 1)
            looped = loop and s in loop_stages
            t_train = int(T.get(s, max(1, L[i] - 1)))
            lo, hi = SHORT_STAGE_RANDOM[s] if (s in SHORT_STAGE_RANDOM and t_train == 1) else _random_range(t_train)
            tk = (4 if hi > 8 else None) if trunc_k == "auto" else trunc_k                        # R7
            self.loop_spec[s] = dict(looped=looped, n_blocks=n_after, t_train=t_train,
                                     t_max=max(2 * t_train, hi), random=(lo, hi), trunc_k=tk)

        # S0 stem (never looped, R11)
        self.stem = nn.Sequential(nn.Conv2d(in_chans, c["stem"], 3, 2, 1, bias=False), nn.BatchNorm2d(c["stem"]),
                                  nn.GELU(), nn.Conv2d(c["stem"], c["stem"], 3, 1, 1))
        res, c_prev = img // 2, c["stem"]
        self.sizes = {}
        for i, s in enumerate(STAGES):
            res //= 2
            size, dim, sp = (res, res), C[i], self.loop_spec[s]
            self.sizes[s] = size
            # transition: unique stride-2 MaxViT block (R1), canonical partitions
            setattr(self, f"s{i + 1}_down", MaxViTBlock(c_prev, dim, size, self.P, self.P, stride=2))
            carry_here = carry and s == "stage3"

            def factory(tm, dim=dim, size=size, looped=sp["looped"], carry_here=carry_here):
                return MaxViTBlock(dim, dim, size, self.P, self.P, t_max=tm, pmal_axes=axes if looped else (),
                                   rel_tables=rel_tables if looped else "auto", carry=carry_here,
                                   masked=window_mask and looped)

            if s == "stage3" and s3_cycle > 1 and sp["looped"]:
                single = factory
                factory = lambda tm: CycleBlock([single(math.ceil(tm / s3_cycle)) for _ in range(s3_cycle)])
            setattr(self, f"s{i + 1}", make_stage(factory, sp["n_blocks"], sp["looped"], dim, sp["t_train"],
                                                  sp["t_max"], sp["trunc_k"], dp(i)))
            c_prev = dim
        self.norm = nn.LayerNorm(C[-1])
        self.head = nn.Linear(C[-1], num_classes)
        self.apply(_init_weights)

    # ------------------------------------------------------------------ helpers
    def random_T_spec(self):
        """{stage: (lo, hi)} for the looped stages (R6)."""
        return {s: sp["random"] for s, sp in self.loop_spec.items() if sp["looped"]}

    def train_T(self):
        return {s: sp["t_train"] for s, sp in self.loop_spec.items() if sp["looped"]}

    def looped_stages(self):
        return {s: getattr(self, "s" + s[-1]) for s in STAGES if isinstance(getattr(self, "s" + s[-1]), LoopedStage)}

    # ------------------------------------------------------------------ forward
    def forward_features(self, x, T=None, return_trace=False):
        T, traces = T or {}, {}
        x = self.stem(x)
        for i, s in enumerate(STAGES):
            x = getattr(self, f"s{i + 1}_down")(x)
            stage = getattr(self, f"s{i + 1}")
            if return_trace:
                x, traces[s] = stage(x, T=T.get(s), return_trace=True)
            else:
                x = stage(x, T=T.get(s))
        feat = self.norm(x.mean((2, 3)))
        return (feat, traces) if return_trace else feat

    def forward(self, x, T=None, return_trace=False):
        if return_trace:
            feat, traces = self.forward_features(x, T, True)
            return self.head(feat), traces
        return self.head(self.forward_features(x, T))


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def build_model(variant="maxvit-t", mode="pmal", img=224, num_classes=1000, carry=False, T=None, drop_path=0.0,
                s3_cycle=1, P="auto", pmal_axes="both", rel_tables="auto", window_mask=False, loop_stages=STAGES,
                **_):
    return LoopMaxViT(variant, img, num_classes, mode=mode, carry=carry, T=T, drop_path=drop_path,
                      s3_cycle=s3_cycle, P=P, pmal_axes=pmal_axes, rel_tables=rel_tables, window_mask=window_mask,
                      loop_stages=tuple(loop_stages or ()))


def model_name(variant, mode, carry=False, s3_cycle=1, pmal_axes="both", rel_tables="auto", window_mask=False, **_):
    tag = {"base": "Base", "shallow": "Shallow", "plain": "LoopPlain", "pmal": "LoopPMAL"}[mode]
    name = f"{variant}-{tag}"
    if mode == "pmal" and pmal_axes != "both":
        name += f"-{pmal_axes}only"
    if window_mask:
        name += "-masked"
    if rel_tables != "auto":
        name += f"-rel{rel_tables.replace('_', '')}"
    return name + ("+M3" if carry else "") + (f"-cycle{s3_cycle}" if s3_cycle > 1 else "")
