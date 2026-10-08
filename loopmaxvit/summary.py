"""summary.py: architecture printout (stage plan, loops, PMAL graphs, params, GMACs), shared by show_arch.py and train.py."""
import logging

import numpy as np
import torch

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)   # "triton not found" is harmless here
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from . import pmal  # noqa: E402
from .model import CFG, STAGES, LoopedStage, build_model, model_name  # noqa: E402

BAR = "=" * 100


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def _device(model):
    return next(model.parameters()).device


def gmacs(model, img, T=None):
    """Forward GMACs for one image. Runs in eval mode so BN running statistics are untouched."""
    was_training = model.training
    model.eval()
    x = torch.zeros(1, 3, img, img, device=_device(model))
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(x, T=T)
    model.train(was_training)
    return fc.get_total_flops() / 2e9


def shapes(model, img):
    """Output shape of every top-level part, via forward hooks."""
    out, hooks = {}, []
    for name, mod in model.named_children():
        hooks.append(mod.register_forward_hook(
            lambda m, i, o, n=name: out.__setitem__(n, tuple((o[0] if isinstance(o, tuple) else o).shape[1:]))))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, img, img, device=_device(model)))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return out


def fmt_shape(s):
    return f"{s[0]}x{s[1]}x{s[2]} map" if len(s) == 3 else f"{s[0]} tok x {s[1]}" if len(s) == 2 else f"{s[0]} logits"


def block_desc(model, looped):
    if not looped:
        return "MaxViT block (fixed partitions)"
    if model.mode == "pmal":
        axes = {"both": "window roll + grid shear", "window": "window roll only", "grid": "grid shear only"}
        d = f"MaxViT block, PMAL ({axes[model.pmal_axes]})"
    else:
        d = "MaxViT block (fixed partitions)"
    return d


def print_plan(model, img):
    c = CFG[model.variant]
    sh = shapes(model, img)
    total = n_params(model)
    print(BAR)
    print(f" {model_name(model.variant, model.mode, model.carry, model.s3_cycle, model.pmal_axes, model.rel_tables, model.window_mask)}"
          f"   input 3x{img}x{img}   classes {model.num_classes}   params {total/1e6:.2f}M")
    print(f" MaxViT config L={c['L']} C={c['C']} stem={c['stem']}, P = G = {model.P}, head dim 32"
          + (f", rel-bias tables {model.rel_tables}" if model.rel_tables != "auto" else "")
          + (", Swin-masked windows (PM3)" if model.window_mask else ""))
    print(BAR)
    print(f" {'stage':5} {'part':10} {'block':52} {'params':>9}   {'output':18} how")
    print("-" * 100)
    rows = [("S0", "stem", "Conv3x3/2 -> BN -> GELU -> Conv3x3", "unique", model.stem)]
    for i, s in enumerate(STAGES):
        down, stage = f"s{i + 1}_down", f"s{i + 1}"
        rows.append((f"S{i + 1}", down, "MaxViT block, MBConv/2 (transition)", "unique", getattr(model, down)))
        st, sp = getattr(model, stage), model.loop_spec[s]
        desc = block_desc(model, sp["looped"]) + (" + carried logits (M3)" if model.carry and s == "stage3" else "")
        if isinstance(st, LoopedStage):
            lo, hi = sp["random"]
            how = f"LOOP x T={sp['t_train']}  (rand-T U[{lo},{hi}], t_max {sp['t_max']}" + \
                  (f", trunc_k {sp['trunc_k']})" if sp["trunc_k"] else ")")
            cyc = f" x{model.s3_cycle} cycled" if s == "stage3" and model.s3_cycle > 1 else ""
            rows.append(("", stage, f"1 shared{cyc} {desc}", how, st))
        else:
            rows.append(("", stage, f"{sp['n_blocks']}x {desc}", "unrolled (distinct)", st))
    rows.append(("head", "norm/head", "mean-pool -> LN -> Linear", "unique", torch.nn.ModuleList([model.norm, model.head])))
    for stg, part, what, how, mod in rows:
        out = fmt_shape(sh[part]) if part in sh else f"{model.num_classes} logits"
        print(f" {stg:5} {part:10} {what:52} {n_params(mod)/1e6:8.2f}M   {out:18} {how}")
    print("-" * 100)


def print_loops(model):
    loops = model.looped_stages()
    if not loops:
        print(" (no looped stages in this mode)\n")
        return
    print(" Looped stages: x_{t+1} = x_t + gamma_t * (B(x~_t; t) - x~_t),   x~_t = x_t + e_t + beta_t * x_0")
    if model.mode == "pmal":
        print("   B(x; t) = GridAttn(pi^g_t) o BlockAttn(pi^w_t) o MBConv(BN_t):  iteration t attends over its own "
              "permuted lattice (M1)")
    for s, st in loops.items():
        shared = n_params(st.block)
        loop_p = st.iter_emb.numel() + st.beta.numel() + st.gamma.numel()
        T = st.t_train
        print(f"   {s}: shared block {shared/1e6:.3f}M params, reused {T}x at T_train "
              f"(+ e/beta/gamma {loop_p:,} params for t_max={st.t_max})")
        seq = " -> ".join(f"B(t={t})" for t in range(min(T, 6))) + (" -> ..." if T > 6 else "")
        print(f"      x0 -> {seq} -> out")
    print()


def print_graphs(model, T=None):
    """PMAL diagnostic (doc 02 sec. 4.2 / risk table): per looped stage, graph overlap of every iteration with t = 0,
    the number of distinct window / grid graphs in T iterations, and the structural mixing time (sec. 4.5)."""
    P = model.P
    print(BAR)
    print(f" PMAL attention graphs (P = G = {P}); overlap = fraction of same-group pairs at t=0 still grouped at t")
    print(BAR)
    for s, st in model.looped_stages().items():
        h, w = model.sizes[s]
        n_iter = (T or {}).get(s, st.t_train)
        n_iter = max(n_iter, 4)
        ident = pmal.identity_perm(h, w)
        pw = [pmal.window_perm(t, h, w, P) for t in range(n_iter)]
        pg = [pmal.grid_perm(t, h, w, P) for t in range(n_iter)]
        ow = [pmal.group_overlap(ident, p, "window", h, w, P) for p in pw]
        og = [pmal.group_overlap(ident, p, "grid", h, w, P) for p in pg]
        nw, ng = pmal.n_distinct_graphs(pw, "window", h, w, P), pmal.n_distinct_graphs(pg, "grid", h, w, P)
        cp = pmal.structural_coverage(h, w, P, P, n_iter, permuted=True)
        cf = pmal.structural_coverage(h, w, P, P, n_iter, permuted=False)
        print(f" {s}: map {h}x{w}, grid stride m = {h // P}, first {n_iter} iterations")
        print(f"    window overlap vs t=0: {np.round(ow, 2).tolist()}   distinct window graphs {nw}")
        print(f"    grid   overlap vs t=0: {np.round(og, 2).tolist()}   distinct grid graphs   {ng}")
        print(f"    coverage (uniform attention rollout from the centre): PMAL {np.round(cp, 2).tolist()}   "
              f"fixed {np.round(cf, 2).tolist()}")
        print(f"    mixing time (90% coverage): PMAL {pmal.mixing_time(cp)}   fixed {pmal.mixing_time(cf)}")
    print()


def print_compare(variant, img, num_classes):
    print(BAR)
    print(f" Mode comparison: {variant}, {img}px, {num_classes} classes  (GMACs at nominal T)")
    print(BAR)
    print(f" {'model':34} {'params':>9} {'vs base':>8} {'GMACs':>7}   T (S1/S2/S3/S4)")
    base = None
    for kw in [dict(mode="base"), dict(mode="shallow"), dict(mode="plain"), dict(mode="plain", carry=True),
               dict(mode="pmal"), dict(mode="pmal", carry=True), dict(mode="pmal", rel_tables="shared")]:
        m = build_model(variant, img=img, num_classes=num_classes, **kw).eval()
        p = n_params(m)
        base = base or p
        T = "/".join(str(m.loop_spec[s]["t_train"]) if m.loop_spec[s]["looped"] else
                     f"{m.loop_spec[s]['n_blocks']}u" for s in STAGES)
        print(f" {model_name(variant, **kw):34} {p/1e6:8.2f}M {p/base-1:+8.0%} {gmacs(m, img):7.2f}   {T}")
    print(" (u = distinct unrolled blocks; looped T = number of passes of one shared block)\n")
