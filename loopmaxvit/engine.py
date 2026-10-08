"""engine.py: config handling, optimizer, schedule and evaluation shared by train.py and evaluate.py."""
import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from .looped_core import recalibrate_bn
from .report import compute_metrics

NO_DECAY_NAMES = ("iter_emb", "beta", "gamma", "alpha", "rel_table")
AMP = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None, None: None}


# ------------------------------------------------------------------ config
def load_config(path, overrides=()):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for ov in overrides:                         # "train.epochs=5"  "model.T={stage3: 4}"
        key, val = ov.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)
    return cfg


# ------------------------------------------------------------------ optimizer / schedule
def param_groups(model, weight_decay):
    """Doc 00 sec. 7: no weight decay on norms, biases and the cheap per-iteration parameters."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or n.split(".")[-1] in NO_DECAY_NAMES:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def lr_at(step, total, warmup, lr, min_lr):
    """Linear warmup, then cosine to min_lr (per step)."""
    if step < warmup:
        return lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * p))


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate(model, loader, T=None, device="cuda", amp_dtype=None, trace=False, keep_outputs=False):
    """Metrics from report.compute_metrics (+ fixed-point residual if trace).
    keep_outputs=True returns (metrics, probs, labels) for write_report."""
    model.eval()
    probs, labels, traces = [], [], {}
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=torch.device(device).type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(x, T=T, return_trace=trace)
        if trace:
            out, tr = out
            for s, v in tr.items():
                traces.setdefault(s, []).append([float(t) for t in v])
        probs.append(F.softmax(out.float(), -1).cpu())
        labels.append(y)
    probs, labels = torch.cat(probs).numpy(), torch.cat(labels).numpy()
    res = compute_metrics(probs, labels, probs.shape[1])
    if trace:  # mean relative update per iteration (fixed-point residual)
        res["residual"] = {s: np.mean(v, 0).round(4).tolist() for s, v in traces.items()}
    return (res, probs, labels) if keep_outputs else res


_streams = {}


def batches(loader, n):
    """Take n batches from one long-lived iterator per loader, restarting it only when exhausted.
    Abandoning a persistent-worker DataLoader mid-epoch on every call is flaky on Windows."""
    for _ in range(n):
        try:
            yield next(_streams[id(loader)])
        except (KeyError, StopIteration):
            _streams[id(loader)] = iter(loader)
            yield next(_streams[id(loader)])


def recal_and_eval(model, recal_loader, val_loader, T, n_recal, device, amp_dtype, trace=False, keep_outputs=False):
    """R13: recalibrate BN at this T, then evaluate at the same T."""
    recalibrate_bn(model, batches(recal_loader, n_recal), T=T, device=device, amp_dtype=amp_dtype)
    return evaluate(model, val_loader, T=T, device=device, amp_dtype=amp_dtype, trace=trace,
                    keep_outputs=keep_outputs)


def acc_vs_T(model, recal_loader, val_loader, n_recal, device, amp_dtype, factor=2, stage="stage3"):
    """H3 / A10: accuracy for T = 1 .. factor * T_train of `stage` (other stages at their T_train).
    Restores the BN statistics afterwards."""
    if stage not in model.train_T():
        return []
    saved = copy.deepcopy({k: v for k, v in model.state_dict().items() if "running" in k or "num_batches" in k})
    t_train, rows = model.train_T()[stage], []
    for t in range(1, factor * t_train + 1):
        T = {**model.train_T(), stage: t}
        r = recal_and_eval(model, recal_loader, val_loader, T, n_recal, device, amp_dtype)
        rows.append({"T": t, **r})
        print(f"   {stage} T={t:2d}: acc {r['acc']:.4f}  macro-F1 {r['macro_f1']:.4f}  ECE {r['ece']:.4f}")
    model.load_state_dict(saved, strict=False)
    return rows


def loop_diagnostics(model):
    """Learned per-iteration parameters: gamma_t / beta_t magnitudes and M3 alpha per head and axis."""
    d = {}
    for s, st in model.looped_stages().items():
        d[s] = {"gamma_mean": st.gamma.detach().mean(1).cpu().round(decimals=4).tolist(),
                "beta_mean": st.beta.detach().mean(1).cpu().round(decimals=4).tolist(),
                "emb_norm": st.iter_emb.detach().norm(dim=1).cpu().round(decimals=4).tolist()}
        for n, p in st.block.named_parameters():
            if n.endswith("alpha"):                                               # (t_max, heads) per axis
                d[s][n.replace(".", "_")] = p.detach().cpu().round(decimals=4).tolist()
    return d
