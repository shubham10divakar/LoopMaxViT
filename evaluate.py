"""
evaluate.py: evaluate a trained run on its own saved split (runs/<run>/split.json) and write a full report.

  python evaluate.py runs/<run>/best.pt                    # test split (val if the run has no test set)
  python evaluate.py runs/<run>/best.pt --split val
  python evaluate.py runs/<run>/best.pt --curve            # + acc-vs-T curve for stage3 (H3 / ablation A10)
  python evaluate.py runs/<run>/best.pt --T stage3=6       # a specific test-time T

Writes runs/<run>/eval/<ckpt>_<split>[_T...]/: metrics.json, per_class.csv, confusion matrices (csv/png/pdf),
classification_report.txt, predictions.csv, reliability diagram. BN statistics are recalibrated at every
evaluated T (rule R13).
"""
import argparse
import json
import os
import sys
from pathlib import Path

# One BLAS thread per process before numpy loads (see train.py): avoids OpenBLAS OOM in Windows workers.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import torch
from torch.utils.data import DataLoader

from loopmaxvit.data import ImageList, load_registry, make_transforms
from loopmaxvit.engine import AMP, acc_vs_T, loop_diagnostics, recal_and_eval
from loopmaxvit.model import build_model
from loopmaxvit.report import fmt_metrics, plot_acc_vs_T, write_report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--split", choices=["test", "val"], help="default: test if the run has one, else val")
    ap.add_argument("--T", nargs="*", help="e.g. stage3=6 stage4=2")
    ap.add_argument("--curve", action="store_true")
    ap.add_argument("--factor", type=int, default=2, help="curve: T = 1 .. factor * T_train")
    ap.add_argument("--recal-batches", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg, ckpt = ck["cfg"], Path(a.checkpoint)
    run = ckpt.parent.parent if ckpt.parent.name == "checkpoints" else ckpt.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = AMP[cfg["train"]["amp"]] if device == "cuda" else None
    classes = ck["classes"]
    model = build_model(num_classes=len(classes), **ck["model_cfg"]).to(device)
    model.load_state_dict(ck["model"])

    split = json.load(open(run / "split.json", encoding="utf-8"))
    which = a.split or ("test" if split.get("test") else "val")
    if not split.get(which):
        raise SystemExit(f"run {run} has no {which} split (it was trained with split {split.get('split', 'train-val')})")
    entry = load_registry()["datasets"][cfg["dataset"]]
    _, t_eval = make_transforms(cfg["model"]["img"], entry.get("mean", [0.485, 0.456, 0.406]),
                                entry.get("std", [0.229, 0.224, 0.225]), cfg["aug"])
    kw = dict(batch_size=a.batch_size, num_workers=a.num_workers, pin_memory=True)
    draft = 2 * cfg["model"]["img"]
    items = [tuple(i) for i in split[which]]
    loader = DataLoader(ImageList(items, t_eval, draft), shuffle=False, **kw)
    recal = DataLoader(ImageList([tuple(i) for i in split["train"]], t_eval, draft), shuffle=True, **kw)

    T = dict(model.train_T())
    for it in a.T or []:
        k, v = it.split("=")
        T[k if k.startswith("stage") else "stage" + k] = int(v)
    print(f"{a.checkpoint}: dataset {cfg['dataset']}, {which} {len(items)} images, epoch {ck['epoch'] + 1}")
    if which == "val":
        print("NOTE: val picked this checkpoint during training; report test numbers in the paper when available")
    r, probs, labels = recal_and_eval(model, recal, loader, T, a.recal_batches, device, amp_dtype,
                                      trace=bool(model.looped_stages()), keep_outputs=True)
    print(f"T {T}: {fmt_metrics(r)}")
    for s, v in r.get("residual", {}).items():
        print(f"   residual ||x_t+1 - x_t|| / ||x_t||  {s}: {v}")

    tag = f"{ckpt.stem}_{which}" + ("".join(f"_{k}={v}" for k, v in T.items()) if a.T else "")
    out_dir = run / "eval" / tag
    write_report(out_dir, probs, labels, classes, items, title=f"{cfg['dataset']} {which} | {ckpt.name}",
                 extra={"checkpoint": str(ckpt), "epoch": ck["epoch"], "split": which, "T": T})
    out = {"checkpoint": str(ckpt), "split": which, "T": T, **r, "loop_params": loop_diagnostics(model)}
    if a.curve:
        out["acc_vs_T_stage3"] = acc_vs_T(model, recal, loader, a.recal_batches, device, amp_dtype, a.factor)
        plot_acc_vs_T(out["acc_vs_T_stage3"], model.train_T().get("stage3", 0), out_dir / "acc_vs_T")
    with open(out_dir / "eval.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"saved {out_dir}")


if __name__ == "__main__":
    main()
