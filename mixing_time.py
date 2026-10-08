"""
mixing_time.py: experiment PM5 (design doc 02 sec. 4.5 / sec. 7), the empirical mixing time of a looped stage.

Attention rollout (Abnar & Zuidema 2020) through the looped stage's block- and grid-attention, iteration by
iteration: every attention becomes an N x N matrix over the stage map (head- and batch-averaged, permuted groups
scattered back to their original positions), mixed with the residual path (0.5 I + 0.5 A). The coverage after
iteration t is the fraction of positions that receive at least `frac` x the uniform share (1/N) of the centre
pixel's influence; the mixing time is the first t with coverage >= 90 %. Only attention is rolled out (the MBConv's
3x3 depthwise conv adds a little local mixing that is ignored, the same for PMAL and fixed partitions).

  python mixing_time.py runs/<pmal run>/best.pt runs/<plain run>/best.pt     # trained checkpoints (val images)
  python mixing_time.py --variant maxvit-t                                   # untrained plain vs PMAL + structure
  python mixing_time.py runs/<run>/best.pt --T 8 --images 256 --frac 0.5

Writes <out>/mixing_time.json and mixing_time.{png,pdf} (default out: runs/<run>/report/ or results/).
"""
import argparse
import json
import os
import sys
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch
from torch.utils.data import DataLoader

from loopmaxvit import pmal
from loopmaxvit.data import ImageList, load_registry, make_transforms
from loopmaxvit.engine import batches
from loopmaxvit.looped_core import LoopedStage, recalibrate_bn
from loopmaxvit.model import build_model, model_name

REPO = Path(__file__).resolve().parent


@torch.no_grad()
def attention_rollout(model, x, stage="stage3", T=None, frac=0.5, centre=None):
    """Per-iteration coverage of the centre pixel's influence through `stage` of `model` on images x."""
    st = model.looped_stages().get(stage)
    assert isinstance(st, LoopedStage), f"{stage} is not looped in this model"
    blk = st.block
    assert hasattr(blk, "block_att"), "rollout supports a single shared MaxViT block (s3_cycle = 1)"
    T = T or st.t_train
    axes = [blk.block_att, blk.grid_att]
    for ax in axes:
        ax.att.record = []
    try:
        model.eval()
        model(x, T={**model.train_T(), stage: T})
        recs = [ax.att.record for ax in axes]
    finally:
        for ax in axes:
            ax.att.record = None
    h, w = model.sizes[stage]
    n = h * w
    mats = []
    for t in range(T):
        ti = min(t, st.t_max - 1)
        for ax, rec in zip(axes, recs):
            probs = rec[t].float().mean(1)                                   # heads -> (B*nG, g, g)
            idx = ax.group_index(ti)                                         # (nG, g) original positions
            probs = probs.view(-1, idx.shape[0], idx.shape[1], idx.shape[1]).mean(0)
            A = torch.zeros(n, n, device=probs.device)
            for g in range(idx.shape[0]):
                A[idx[g][:, None], idx[g][None, :]] = probs[g]
            mats.append(A.cpu().double().numpy())
    centre = (h // 2) * w + w // 2 if centre is None else centre
    return pmal.rollout_coverage(mats, centre, frac)[1::2]                   # after each full block


def load_run(ckpt_path, device, n_images, n_recal, T, stage):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    run = Path(ckpt_path).parent
    run = run.parent if run.name == "checkpoints" else run
    model = build_model(num_classes=len(ck["classes"]), **ck["model_cfg"]).to(device)
    model.load_state_dict(ck["model"])
    split = json.load(open(run / "split.json", encoding="utf-8"))
    entry = load_registry()["datasets"][cfg["dataset"]]
    _, t_eval = make_transforms(cfg["model"]["img"], entry.get("mean"), entry.get("std"), cfg["aug"])
    draft = 2 * cfg["model"]["img"]
    val = DataLoader(ImageList([tuple(i) for i in split["val"]][:n_images], t_eval, draft), batch_size=n_images)
    recal = DataLoader(ImageList([tuple(i) for i in split["train"]], t_eval, draft), batch_size=64, shuffle=True)
    T_eval = {**model.train_T(), stage: T or model.train_T()[stage]}
    recalibrate_bn(model, batches(recal, n_recal), T=T_eval, device=device)   # R13 at the evaluated T
    x, _ = next(iter(val))
    return model, x.to(device), run, model_name(**ck["model_cfg"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="*", help="best.pt / last.pt of looped runs (empty: untrained models)")
    ap.add_argument("--variant", default="maxvit-t", help="untrained mode only")
    ap.add_argument("--img", type=int, default=224, help="untrained mode only")
    ap.add_argument("--stage", default="stage3")
    ap.add_argument("--T", type=int, help="iterations to roll out (default 2 x T_train)")
    ap.add_argument("--images", type=int, default=64)
    ap.add_argument("--recal-batches", type=int, default=20)
    ap.add_argument("--frac", type=float, default=0.5, help="covered = influence >= frac / N")
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--out", help="output directory")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    rows, out_dir = [], Path(a.out) if a.out else None
    if a.checkpoints:
        for c in a.checkpoints:
            model, x, run, name = load_run(c, device, a.images, a.recal_batches, a.T, a.stage)
            out_dir = out_dir or run / "report"
            rows.append((f"{name} ({run.name})", model, x))
    else:
        out_dir = out_dir or REPO / "results"
        x = torch.randn(a.images, 3, a.img, a.img, device=device)
        for mode in ("plain", "pmal"):
            rows.append((f"{model_name(a.variant, mode)} (untrained)", build_model(a.variant, mode, a.img, 10).to(device), x))

    results = {}
    for label, model, x in rows:
        T = a.T or 2 * model.train_T()[a.stage]
        cov = attention_rollout(model, x, a.stage, T, a.frac)
        results[label] = {"coverage": [round(c, 4) for c in cov], "mixing_time": pmal.mixing_time(cov, a.target),
                          "T": T, "mode": model.mode}
        print(f"{label:60s} mixing time {results[label]['mixing_time']}   coverage {np.round(cov, 3).tolist()}")
    # the structural reference: uniform attention inside each group, same geometry
    model = rows[0][1]
    h, w = model.sizes[a.stage]
    T = max(r["T"] for r in results.values())
    for permuted in (False, True):
        cov = pmal.structural_coverage(h, w, model.P, model.P, T, permuted=permuted, frac=a.frac)
        label = f"structure only, {'PMAL' if permuted else 'fixed partitions'}"
        results[label] = {"coverage": [round(c, 4) for c in cov], "mixing_time": pmal.mixing_time(cov, a.target), "T": T}
        print(f"{label:60s} mixing time {results[label]['mixing_time']}   coverage {np.round(cov, 3).tolist()}")

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"stage": a.stage, "map": [h, w], "P": model.P, "frac": a.frac, "target": a.target, "images": a.images}
    with open(out_dir / "mixing_time.json", "w", encoding="utf-8") as f:
        json.dump({**meta, "results": results}, f, indent=2)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, r in results.items():
        ax.plot(range(1, len(r["coverage"]) + 1), r["coverage"], "--" if label.startswith("structure") else "o-",
                label=label, lw=1 if label.startswith("structure") else 1.8)
    ax.axhline(a.target, color="gray", ls=":", lw=1)
    ax.set_xlabel(f"{a.stage} iteration t")
    ax.set_ylabel(f"coverage (influence >= {a.frac}/N)")
    ax.set_title(f"Mixing time (PM5), {a.stage} map {h}x{w}, P = G = {model.P}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"mixing_time.{ext}", dpi=300, bbox_inches="tight")
    print(f"saved {out_dir / 'mixing_time.json'} and mixing_time.png/pdf")


if __name__ == "__main__":
    main()
