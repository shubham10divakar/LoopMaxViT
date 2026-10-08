"""
train.py: train LoopMaxViT on a registered dataset (images read in place, nothing copied).

  python train.py                                              # configs/default.yaml (paddy, maxvit-t PMAL)
  python train.py --dataset plantvillage                       # 80/10/10 train/val/test (default)
  python train.py --dataset plantvillage --split train-val     # 80/20 train/val only, no test set
  python train.py --dataset plantwild --official-split         # official train / val / test
  python train.py --set train.epochs=5 train.limit=512         # quick sanity run
  python train.py --resume runs/<run>/last.pt                  # or runs/<run>/checkpoints/epoch_XXX.pt
  python train.py --mode plain                                 # Loop-plain (fixed partitions), ablation PM1
  python train.py --pmal-axes grid --rel-tables shared         # ablations PM2 / PM4 (--window-mask: PM3)
  python train.py --dataset plantvillage --fresh               # start over even if the run exists

Rerunning the same command continues an unfinished run from its last.pt automatically; a finished run
(final.json present) is not retrained unless --fresh is given.

Splits: train trains; val picks best.pt and drives early stopping; test is evaluated once, at the end.
With --split train-val there is no test set and the final report is on val (optimistic: val also picked best.pt).

Outputs in runs/<run>/:
  config.yaml  split.json  model_summary.txt  log.csv  best.pt  last.pt  checkpoints/epoch_XXX.pt  final.json
  report/training_curves.{png,pdf}  report/acc_vs_T.{png,pdf}
  report/<ckpt>_<split>/  metrics.json  per_class.csv  confusion_matrix*.{csv,png,pdf}  predictions.csv  ...
"""
import argparse
import contextlib
import csv
import io
import json
import os
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

# One BLAS thread per process, set before numpy loads. On Windows every DataLoader worker is a fresh
# process; with per-core OpenBLAS buffers in each one, 8 workers exhaust the commit limit
# ("OpenBLAS error: Memory allocation still failed"). Workers inherit these.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch
import torch.nn as nn
import yaml
from torchvision.transforms import v2

from loopmaxvit.data import build_loaders, default_val_frac
from loopmaxvit.engine import (AMP, acc_vs_T, load_config, loop_diagnostics, lr_at, param_groups,
                                recal_and_eval)
from loopmaxvit.looped_core import sample_T
from loopmaxvit.model import MODES, build_model, model_name
from loopmaxvit.report import (HEADLINE, fmt_metrics, plot_acc_vs_T, plot_training_curves, throughput,
                                write_report)
from loopmaxvit.summary import BAR, gmacs, n_params, print_graphs, print_loops, print_plan

REPO = Path(__file__).resolve().parent
MONITOR = {"val_acc": ("acc", "max"), "val_macro_f1": ("macro_f1", "max"), "val_loss": ("nll", "min")}
LOG_FIELDS = ["epoch", "lr", "train_loss", "train_acc", "val_nll", "val_acc", "val_macro_f1", "val_ece", "mean_T3",
              "time_s", "best"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "configs" / "default.yaml"))
    ap.add_argument("--dataset")
    ap.add_argument("--data-root")
    ap.add_argument("--variant")
    ap.add_argument("--mode", choices=list(MODES), help="base | shallow | plain (Loop-plain) | pmal (Loop-full, M1)")
    ap.add_argument("--carry", action="store_true", help="M3 per-axis loop-carried attention logits in S3")
    ap.add_argument("--pmal-axes", choices=["both", "window", "grid"], help="PM2: which axes PMAL permutes")
    ap.add_argument("--window-mask", action="store_true", help="PM3: Swin-masked instead of cyclic windows")
    ap.add_argument("--rel-tables", choices=["auto", "shared", "per_iter"], help="PM4: relative-bias tables")
    ap.add_argument("--P", help="window = grid size; auto = img/32")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--img", type=int)
    ap.add_argument("--split", choices=["train-val-test", "train-val"],
                    help="train-val-test (80/10/10, default) or train-val (80/20, no test)")
    ap.add_argument("--val-frac", type=float, help="fraction of the whole pool (default 0.1 / 0.2)")
    ap.add_argument("--test-frac", type=float, help="train-val-test only (default 0.1)")
    ap.add_argument("--official-split", action="store_true", help="plantwild: official train / val / test")
    ap.add_argument("--patience", type=int, help="early-stopping patience in epochs, 0 = off")
    ap.add_argument("--monitor", choices=list(MONITOR), help="early-stopping / best.pt metric")
    ap.add_argument("--save-every", type=int, help="epoch checkpoint every N epochs, 0 = off")
    ap.add_argument("--keep-checkpoints", type=int, help="keep newest N epoch checkpoints, 0 = all")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--name", help="run name (default: <dataset>_<model>_<split>_s<seed>)")
    ap.add_argument("--resume", help="path to last.pt or checkpoints/epoch_XXX.pt (default: the run's own last.pt, "
                                     "if an unfinished run with the same name exists)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore an existing run with the same name and start over (it is kept, renamed *_old_<time>)")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. train.lr=1e-3 model.T={stage3: 4}")
    return ap.parse_args()


def apply_train_cli(cfg, a):
    """Options that may also change on --resume (they do not change the data or the model)."""
    tc, es = cfg["train"], cfg["train"]["early_stopping"]
    if a.epochs is not None:
        tc["epochs"] = a.epochs
    if a.patience is not None:
        es["patience"] = a.patience
    if a.monitor is not None:
        es["metric"] = a.monitor
    if a.save_every is not None:
        tc["save_every"] = a.save_every
    if a.keep_checkpoints is not None:
        tc["keep_checkpoints"] = a.keep_checkpoints
    return cfg


def apply_cli(cfg, a):
    for key, val in [("dataset", a.dataset), ("data_root", a.data_root), ("seed", a.seed), ("split", a.split),
                     ("val_frac", a.val_frac), ("test_frac", a.test_frac)]:
        if val is not None:
            cfg[key] = val
    for key, val in [("variant", a.variant), ("mode", a.mode), ("img", a.img), ("pmal_axes", a.pmal_axes),
                     ("rel_tables", a.rel_tables), ("P", a.P)]:
        if val is not None:
            cfg["model"][key] = val
    if a.carry:
        cfg["model"]["carry"] = True
    if a.window_mask:
        cfg["model"]["window_mask"] = True
    if a.official_split:
        cfg["official_split"] = True
    if a.batch_size is not None:
        cfg["train"]["batch_size"] = a.batch_size
    if cfg.get("val_frac") is None:
        cfg["val_frac"] = default_val_frac(cfg["split"])
    return apply_train_cli(cfg, a)


def merge_defaults(cfg, defaults):
    """Checkpoints from older versions of this script lack the newer keys; fill them from the defaults."""
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v
        elif isinstance(v, dict) and isinstance(cfg[k], dict):
            merge_defaults(cfg[k], v)
    return cfg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def atomic_save(obj, path):
    """Write to a temp file, then rename: an interrupted save never corrupts last.pt."""
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def run_header(cfg, dl, model, name, run, device, amp_dtype, lr, steps_per_epoch, warm_ep):
    """Everything about the run, printed at the start and saved to model_summary.txt."""
    tc, mc, ac, es = cfg["train"], cfg["model"], cfg["aug"], cfg["train"]["early_stopping"]
    classes = dl["classes"]
    sizes = {s: len(dl[f"{s}_items"]) for s in ("train", "val", "test")}
    total = sum(sizes.values())
    counts = Counter(y for _, y in dl["train_items"])
    lo, hi = min(counts.values()), max(counts.values())
    print(BAR)
    print(f" run      {run}")
    print(f" device   {torch.cuda.get_device_name() if device == 'cuda' else 'cpu'}   torch {torch.__version__}   "
          f"amp {tc['amp'] if amp_dtype else 'off'}")
    print(f" dataset  {cfg['dataset']}: {len(classes)} classes, {total:,} images"
          + ("  (official split)" if cfg["official_split"] else ""))
    print(f" split    {cfg['split']}: " + "  ".join(f"{s} {n:,} ({n / total:.0%})" for s, n in sizes.items() if n)
          + f"   stratified, seed {cfg['seed']}")
    print(f"          train images per class: min {lo} ({classes[min(counts, key=counts.get)]}), "
          f"max {hi} ({classes[max(counts, key=counts.get)]}), imbalance {hi / lo:.1f}x")
    if cfg["split"] == "train-val":
        print("          NOTE: no test set; final numbers are on val, which also picks best.pt (optimistic)")
    print(f" optim    AdamW lr {lr:.2e} (base {tc['base_lr']} x {tc['batch_size']}/512), wd {tc['weight_decay']}, "
          f"cosine to {tc['min_lr']}, warmup {warm_ep} ep, grad-clip {tc['grad_clip']}, "
          f"label-smoothing {tc['label_smoothing']}")
    print(f" schedule {tc['epochs']} epochs x {steps_per_epoch} steps, batch {tc['batch_size']}, "
          f"random-T {'on' if tc['random_T'] else 'off'}")
    print(f" stopping monitor {es['metric']}, patience "
          + (f"{es['patience']} epochs (after warmup), min_delta {es['min_delta']}" if es["patience"] else "off"))
    print(f" ckpts    best.pt + last.pt every epoch" + (
        f", checkpoints/epoch_XXX.pt every {tc['save_every']} (keep "
        f"{tc['keep_checkpoints'] or 'all'})" if tc["save_every"] else ""))
    print(f" aug      crop {ac.get('crop_scale')}, vflip {ac.get('vflip')}, RandAugment {ac.get('randaugment')}, "
          f"erasing {ac.get('random_erasing')}, mixup {ac.get('mixup')}, cutmix {ac.get('cutmix')} "
          f"(p {ac.get('mix_prob')}), {mc['img']}px")
    print_plan(model, mc["img"])
    print_loops(model)
    if model.looped_stages():
        print_graphs(model)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" {name}: {n_params(model) / 1e6:.2f}M params ({n_train / 1e6:.2f}M trainable), "
          f"{gmacs(model, mc['img']):.2f} GMACs at nominal T {model.train_T() or '-'}")
    print(BAR)


def read_log(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_log(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    a = parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    defaults = load_config(REPO / "configs" / "default.yaml")
    cli_cfg = load_config(a.config, a.set)
    if not a.resume:
        cfg = apply_cli(cli_cfg, a)
        mc = cfg["model"]
        name = model_name(**mc)
        run = REPO / cfg["out_dir"] / (a.name or f"{cfg['dataset']}_{name}_{cfg['split']}_s{cfg['seed']}")
        if (run / "last.pt").exists() and a.fresh:
            old = run.with_name(f"{run.name}_old_{time.strftime('%Y%m%d-%H%M%S')}")
            run.rename(old)
            print(f"--fresh: previous run moved to {old}")
        elif (run / "final.json").exists():
            print(f"run {run.name} already finished (final.json exists); results are in {run}\n"
                  f"  use --fresh to train again from scratch (the old run is kept, renamed), or --name for a new run")
            return
        elif (run / "last.pt").exists():
            a.resume = str(run / "last.pt")
            print(f"found an unfinished run in {run}: resuming from last.pt (use --fresh to start over)")
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        resumed_epochs = cfg["train"]["epochs"]
        if "split" not in cfg:               # runs from before the 3-way split: 2-way pool split
            cfg["split"] = "train-val"
        cfg = apply_train_cli(merge_defaults(cfg, defaults), a)
        # worker count does not change results; take it from the current config/--set (e.g. to fix memory issues)
        cfg["train"]["num_workers"] = cli_cfg["train"]["num_workers"]
        print("  the run's saved config is used; only --epochs/--patience/--monitor/--save-every/"
              "--keep-checkpoints and train.num_workers can change on resume")
    tc, mc, ac, es = cfg["train"], cfg["model"], cfg["aug"], cfg["train"]["early_stopping"]
    mon_key, mon_mode = MONITOR[es["metric"]]
    better = (lambda v, b: v > b + es["min_delta"]) if mon_mode == "max" else (lambda v, b: v < b - es["min_delta"])
    seed_all(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = AMP[tc["amp"]] if device == "cuda" else None
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # ---------------------------------------------------------------- data
    t0 = time.time()
    dl = build_loaders(cfg["dataset"], mc["img"], tc["batch_size"], ac, data_root=cfg["data_root"],
                       split=cfg["split"], val_frac=cfg["val_frac"], test_frac=cfg["test_frac"], seed=cfg["seed"],
                       official_split=cfg["official_split"], num_workers=tc["num_workers"], limit=tc.get("limit"))
    classes = dl["classes"]
    print(f"indexed {cfg['dataset']} in {time.time() - t0:.1f}s")

    # ---------------------------------------------------------------- model / run dir
    model = build_model(num_classes=len(classes), **mc).to(device)
    name = model_name(**mc)
    if a.resume:
        run = Path(a.resume).parent
        run = run.parent if run.name == "checkpoints" else run
    else:
        run = REPO / cfg["out_dir"] / (a.name or f"{cfg['dataset']}_{name}_{cfg['split']}_s{cfg['seed']}")
    run.mkdir(parents=True, exist_ok=True)
    (run / "report").mkdir(exist_ok=True)
    n_par = n_params(model)

    lr = tc["base_lr"] * tc["batch_size"] / 512
    opt = torch.optim.AdamW(param_groups(model, tc["weight_decay"]), lr=lr, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler(enabled=amp_dtype == torch.float16)
    crit = nn.CrossEntropyLoss(label_smoothing=tc["label_smoothing"])
    mixers = [m(alpha=al, num_classes=len(classes))
              for m, al in ((v2.MixUp, ac.get("mixup", 0)), (v2.CutMix, ac.get("cutmix", 0))) if al > 0]
    mixer = v2.RandomChoice(mixers) if mixers and ac.get("mix_prob", 0) > 0 else None

    steps_per_epoch = len(dl["train"])
    warm_ep = min(tc["warmup_epochs"], max(1, tc["epochs"] // 5))
    total, warmup = tc["epochs"] * steps_per_epoch, warm_ep * steps_per_epoch

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_header(cfg, dl, model, name, run, device, amp_dtype, lr, steps_per_epoch, warm_ep)
    print(buf.getvalue(), end="")
    try:                                         # full layer table only goes to the file
        from torchinfo import summary
        buf.write("\n" + str(summary(model, input_size=(1, 3, mc["img"], mc["img"]), depth=4, verbose=0,
                                     col_names=("output_size", "num_params"), row_settings=("var_names",),
                                     device=device, mode="eval")) + "\n")
    except Exception as e:                       # torchinfo is optional
        buf.write(f"\n(torchinfo summary unavailable: {e})\n")
    buf.write("\n" + str(model) + "\n")
    (run / "model_summary.txt").write_text(buf.getvalue(), encoding="utf-8")

    start_epoch, best, best_epoch, bad_epochs, step = 0, None, -1, 0, 0
    log_path = run / "log.csv"
    if a.resume:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        if "rng" in ck:
            set_rng_state(ck["rng"])
        start_epoch = ck["epoch"] + 1
        best, best_epoch = ck["best"], ck.get("best_epoch", -1)
        bad_epochs, step = ck.get("bad_epochs", 0), ck.get("step", start_epoch * steps_per_epoch)
        write_log(log_path, [r for r in read_log(log_path) if int(r["epoch"]) <= ck["epoch"]])
        print(f"resumed from {a.resume}: continuing at epoch {start_epoch + 1}/{tc['epochs']}, best {es['metric']} {best} "
              f"(epoch {best_epoch + 1}), {bad_epochs} epochs without improvement")
        if resumed_epochs != tc["epochs"]:
            print("  NOTE: epochs changed on resume; the cosine LR schedule is stretched to the new length")
    else:
        with open(run / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        with open(run / "split.json", "w", encoding="utf-8") as f:
            json.dump({"split": cfg["split"], "classes": classes, "train": dl["train_items"],
                       "val": dl["val_items"], "test": dl["test_items"]}, f)
        write_log(log_path, [])

    def checkpoint(epoch):
        return {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "best": best, "best_epoch": best_epoch, "bad_epochs": bad_epochs, "step": step,
                "rng": rng_state(), "cfg": cfg, "model_cfg": mc, "classes": classes, "dataset": cfg["dataset"]}

    # live in-epoch progress, rewritten in place; only on a terminal so redirected logs stay clean
    live = sys.stdout.isatty()

    def status(msg, _w=[0]):
        sys.stdout.write("\r" + msg.ljust(_w[0]) + ("\r" if not msg else ""))
        sys.stdout.flush()
        _w[0] = len(msg)

    # ---------------------------------------------------------------- train
    stopped_early = False
    for epoch in range(start_epoch, tc["epochs"]):
        if es["patience"] and bad_epochs >= es["patience"]:
            stopped_early = True
            break
        model.train()
        t0, tot_loss, correct, seen, sumT3, nb = time.time(), 0.0, 0, 0, 0, 0
        for x, y in dl["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            y_hard = y
            if mixer is not None and random.random() < ac["mix_prob"]:
                x, y = mixer(x, y)
            T = sample_T(model.random_T_spec()) if tc["random_T"] else None
            sumT3 += (T or model.train_T()).get("stage3", 0)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, warmup, lr, tc["min_lr"])
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(x, T=T)
                loss = crit(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            step += 1
            nb += 1
            tot_loss += loss.item() * x.shape[0]
            correct += (logits.argmax(1) == y_hard).sum().item()
            seen += x.shape[0]
            if live and (nb % 10 == 0 or nb == steps_per_epoch):
                el = time.time() - t0
                eta = el / nb * (steps_per_epoch - nb)
                status(f"ep {epoch + 1:3d}/{tc['epochs']}  step {nb:4d}/{steps_per_epoch}  "
                       f"loss {tot_loss / seen:.4f}  train-acc {correct / seen:.4f}  "
                       f"lr {opt.param_groups[0]['lr']:.2e}  {el:.0f}s elapsed, ~{eta:.0f}s left")
        if live:
            status(f"ep {epoch + 1:3d}/{tc['epochs']}  validating ...")
        # R13: recalibrate BN at the training T, then validate
        r = recal_and_eval(model, dl["recal"], dl["val"], model.train_T(), tc["recal_batches_epoch"],
                           device, amp_dtype)
        if live:
            status("")
        dt = time.time() - t0
        flag = ""
        if best is None or better(r[mon_key], best):
            best, best_epoch, bad_epochs, flag = r[mon_key], epoch, 0, "  *best"
        elif epoch + 1 > warm_ep:
            bad_epochs += 1
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(dict(
                epoch=epoch, lr=f"{opt.param_groups[0]['lr']:.2e}", train_loss=f"{tot_loss / seen:.4f}",
                train_acc=f"{correct / seen:.4f}", val_nll=f"{r['nll']:.4f}", val_acc=f"{r['acc']:.4f}",
                val_macro_f1=f"{r['macro_f1']:.4f}", val_ece=f"{r['ece']:.4f}", mean_T3=f"{sumT3 / max(1, nb):.2f}",
                time_s=f"{dt:.0f}", best=int(bool(flag))))
        atomic_save(checkpoint(epoch), run / "last.pt")
        if flag:
            shutil.copyfile(run / "last.pt", run / "best.pt")
        if tc["save_every"] and (epoch + 1) % tc["save_every"] == 0:
            (run / "checkpoints").mkdir(exist_ok=True)
            shutil.copyfile(run / "last.pt", run / "checkpoints" / f"epoch_{epoch + 1:03d}.pt")
            if tc["keep_checkpoints"]:
                for old in sorted((run / "checkpoints").glob("epoch_*.pt"))[:-tc["keep_checkpoints"]]:
                    old.unlink()
        stop = f"  (no improvement {bad_epochs}/{es['patience']})" if es["patience"] and bad_epochs else ""
        print(f"ep {epoch + 1:3d}/{tc['epochs']}  loss {tot_loss / seen:.4f}  train-acc {correct / seen:.4f}  "
              f"val-acc {r['acc']:.4f}  F1 {r['macro_f1']:.4f}  NLL {r['nll']:.4f}  ({dt:.0f}s){flag}{stop}")
        if es["patience"] and bad_epochs >= es["patience"]:
            stopped_early = True
            print(f"early stopping: {es['metric']} has not improved for {es['patience']} epochs "
                  f"(best {best:.4f} at epoch {best_epoch + 1})")
            break

    # ---------------------------------------------------------------- final evaluation
    log_rows = read_log(log_path)
    epochs_run = int(log_rows[-1]["epoch"]) + 1 if log_rows else 0
    stopped_early = stopped_early or epochs_run < tc["epochs"]
    eval_split = "test" if dl["test"] is not None else "val"
    print(f"\nfinal evaluation on {eval_split} (BN recalibrated at each T, rule R13)")
    if eval_split == "val":
        print("  NOTE: split train-val has no test set; val also picked best.pt, so these numbers are optimistic")
    jobs = [("last", eval_split)] + ([("best", "val")] if eval_split != "val" else []) + [("best", eval_split)]
    results, looped = {}, bool(model.looped_stages())
    for tag, split in jobs:
        sd = torch.load(run / f"{tag}.pt", map_location=device, weights_only=False)
        model.load_state_dict(sd["model"])
        m, probs, labels = recal_and_eval(model, dl["recal"], dl[split], model.train_T(), tc["recal_batches"],
                                          device, amp_dtype, trace=looped, keep_outputs=True)
        key = f"{tag}_{split}"
        write_report(run / "report" / key, probs, labels, classes, dl[f"{split}_items"],
                     title=f"{name} | {cfg['dataset']} {split} | {tag}.pt (epoch {sd['epoch'] + 1})",
                     extra={"checkpoint": f"{tag}.pt", "epoch": sd["epoch"], "split": split, "T": model.train_T()})
        results[key] = {"epoch": sd["epoch"], **m}
        print(f"  {tag}.pt (epoch {sd['epoch'] + 1:3d}) on {split:5s}: {fmt_metrics(m)}")
    headline = results[f"best_{eval_split}"]
    if headline.get("residual"):
        print(f"  fixed-point residual per iteration: {headline['residual']}")

    # model is now best.pt
    print(f"\naccuracy vs test-time T on {eval_split} (best.pt)")
    curve = acc_vs_T(model, dl["recal"], dl[eval_split], tc["recal_batches"], device, amp_dtype,
                     tc["eval_T_max_factor"])
    plot_acc_vs_T(curve, model.train_T().get("stage3", 0), run / "report" / "acc_vs_T")
    plot_training_curves(log_path, run / "report" / "training_curves", best_epoch)
    eff = {"params": n_par, "params_M": round(n_par / 1e6, 3), "gmacs": round(gmacs(model, mc["img"]), 3),
           "train_time_h": round(sum(float(r["time_s"]) for r in log_rows) / 3600, 3)}
    if tc.get("benchmark", True):
        eff.update(throughput(model, mc["img"], device, amp_dtype))

    out = {"model": name, "dataset": cfg["dataset"], "variant": mc["variant"], "mode": mc["mode"],
           "carry": mc.get("carry", False), "pmal_axes": mc.get("pmal_axes", "both"),
           "rel_tables": mc.get("rel_tables", "auto"), "window_mask": mc.get("window_mask", False),
           "P": model.P, "seed": cfg["seed"], "split": cfg["split"],
           "official_split": cfg["official_split"],
           "split_sizes": {s: len(dl[f"{s}_items"]) for s in ("train", "val", "test")},
           "eval_split": eval_split, "selection_metric": es["metric"], "best_val": best,
           "best_epoch": best_epoch, "epochs_run": epochs_run, "epochs_max": tc["epochs"],
           "stopped_early": stopped_early, "headline": headline, "results": results,
           "acc_vs_T_stage3": curve, "efficiency": eff, "loop_params": loop_diagnostics(model)}
    with open(run / "final.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n" + BAR)
    print(f" {name} on {cfg['dataset']}  |  best epoch {best_epoch + 1}/{epochs_run}"
          + ("  (early stopped)" if stopped_early else ""))
    print(f" {eval_split.upper()} (best.pt):")
    for k in HEADLINE:
        if headline.get(k) is not None:
            print(f"   {k:16s} {headline[k]:.4f}")
    print(f" params {eff['params_M']}M  GMACs {eff['gmacs']}"
          + (f"  throughput {eff['throughput_img_s']} img/s  latency {eff['latency_ms_bs1']} ms" if "throughput_img_s"
             in eff else ""))
    print(f" reports: {run / 'report'}   summary: {run / 'final.json'}")
    print(BAR)


if __name__ == "__main__":
    main()
