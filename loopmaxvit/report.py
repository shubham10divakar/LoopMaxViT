"""report.py: paper metrics, confusion matrices, per-class reports, plots and efficiency numbers.

compute_metrics() is used every epoch (cheap); write_report() runs once per final evaluation and writes
    metrics.json  per_class.csv  classification_report.txt  predictions.csv
    confusion_matrix.csv  confusion_matrix_normalized.csv  confusion_matrix{,_normalized}.{png,pdf}
    per_class_f1.{png,pdf}  reliability.{png,pdf}
"""
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (balanced_accuracy_score, classification_report, cohen_kappa_score, confusion_matrix,
                             matthews_corrcoef, precision_recall_fscore_support, roc_auc_score)

# metrics that go into the paper table (aggregate.py) and the console summary, in this order
HEADLINE = ("acc", "balanced_acc", "top5_acc", "precision_macro", "recall_macro", "macro_f1", "weighted_f1",
            "cohen_kappa", "mcc", "roc_auc_macro", "ece", "nll")


# ------------------------------------------------------------------ metrics
def ece_score(probs, labels, n_bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == labels).astype(float)
    edges, e = np.linspace(0, 1, n_bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def compute_metrics(probs, labels, num_classes):
    """probs (N, C) softmax, labels (N,). Every value is a plain float (None if undefined)."""
    probs = probs.astype(np.float64)
    probs = probs / probs.sum(1, keepdims=True)  # float32 softmax rows can miss 1.0 enough for roc_auc_score
    pred = probs.argmax(1)
    k = min(5, num_classes)
    topk = np.argsort(-probs, 1)[:, :k]
    cls = list(range(num_classes))
    pm, rm, fm, _ = precision_recall_fscore_support(labels, pred, labels=cls, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(labels, pred, labels=cls, average="weighted", zero_division=0)
    auc = None
    if len(np.unique(labels)) == num_classes:   # one-vs-rest AUC is undefined for a class with no samples
        auc = roc_auc_score(labels, probs[:, 1]) if num_classes == 2 else \
            roc_auc_score(labels, probs, multi_class="ovr", average="macro", labels=cls)
    return dict(
        acc=float((pred == labels).mean()),
        balanced_acc=float(balanced_accuracy_score(labels, pred)),
        top5_acc=float((topk == labels[:, None]).any(1).mean()),
        precision_macro=float(pm), recall_macro=float(rm), macro_f1=float(fm),
        precision_weighted=float(pw), recall_weighted=float(rw), weighted_f1=float(fw),
        cohen_kappa=float(cohen_kappa_score(labels, pred)),
        mcc=float(matthews_corrcoef(labels, pred)),
        roc_auc_macro=None if auc is None else float(auc),
        ece=ece_score(probs, labels),
        nll=float(-np.log(probs[np.arange(len(labels)), labels] + 1e-12).mean()),
        n=int(len(labels)))


def fmt_metrics(m):
    return "  ".join(f"{k} {m[k]:.4f}" for k in HEADLINE if m.get(k) is not None)


# ------------------------------------------------------------------ plots
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    _plt().close(fig)


def short_name(c, width=28):
    c = c.replace("___", ": ").replace("_", " ")
    return c if len(c) <= width else c[:width - 1] + "…"


def plot_confusion(cm, classes, stem, normalize, title=""):
    plt = _plt()
    C = len(classes)
    data = cm / cm.sum(1, keepdims=True).clip(min=1) if normalize else cm
    side = max(5.0, 0.32 * C + 3)
    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1 if normalize else max(1, data.max()))
    names = [short_name(c) for c in classes]
    ax.set_xticks(range(C), names, rotation=90 if C > 6 else 45, ha="center" if C > 6 else "right",
                  fontsize=max(5, 10 - C // 8))
    ax.set_yticks(range(C), names, fontsize=max(5, 10 - C // 8))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title + (" (row-normalized)" if normalize else ""))
    if C <= 40:
        fs, hi = max(3.5, 9 - C // 6), (1 if normalize else data.max()) / 2
        for i in range(C):
            for j in range(C):
                v = data[i, j]
                if v == 0 and i != j:
                    continue
                txt = (f"{v:.2f}" if C <= 15 else f"{100 * v:.0f}") if normalize else f"{int(v)}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=fs, color="white" if v > hi else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, stem)


def plot_per_class_f1(rows, stem, title=""):
    plt = _plt()
    names = [short_name(r["class"], 40) for r in rows]
    f1 = [r["f1"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.22 * len(rows) + 1)))
    ax.barh(range(len(rows)), f1, color="#4C72B0")
    ax.set_yticks(range(len(rows)), names, fontsize=max(5, 9 - len(rows) // 10))
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    _save(fig, stem)


def plot_reliability(probs, labels, stem, n_bins=15, title=""):
    plt = _plt()
    conf, acc = probs.max(1), (probs.argmax(1) == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    centers, accs, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        centers.append((lo + hi) / 2)
        accs.append(acc[m].mean() if m.any() else 0)
        counts.append(m.sum())
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.bar(centers, accs, width=1 / n_bins, edgecolor="black", color="#4C72B0", label="accuracy")
    ax.bar(centers, np.array(centers) - np.array(accs), bottom=accs, width=1 / n_bins, color="#DD8452",
           alpha=0.5, edgecolor="#DD8452", label="gap")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title}  ECE {ece_score(probs, labels, n_bins):.4f}")
    ax.legend(loc="upper left")
    _save(fig, stem)


def plot_training_curves(log_csv, stem, best_epoch=None):
    """log.csv -> loss / accuracy / val F1 / LR panels."""
    with open(log_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    plt = _plt()
    ep = [int(r["epoch"]) + 1 for r in rows]
    col = lambda k: [float(r[k]) if r.get(k) not in (None, "") else np.nan for r in rows]   # noqa: E731
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    panels = [(axs[0, 0], "Loss", [("train_loss", "train"), ("val_nll", "val (NLL)")]),
              (axs[0, 1], "Accuracy", [("train_acc", "train"), ("val_acc", "val")]),
              (axs[1, 0], "Validation macro-F1", [("val_macro_f1", "val")]),
              (axs[1, 1], "Learning rate", [("lr", "lr")])]
    for ax, title, series in panels:
        for k, lab in series:
            if k in rows[0]:
                ax.plot(ep, col(k), label=lab)
        if best_epoch is not None:
            ax.axvline(best_epoch + 1, color="gray", ls="--", lw=1, label="best")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axs[1, 1].set_yscale("log")
    fig.tight_layout()
    _save(fig, stem)


def plot_acc_vs_T(curve, t_train, stem, stage="stage3"):
    if not curve:
        return
    plt = _plt()
    T = [r["T"] for r in curve]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(T, [r["acc"] for r in curve], "o-", label="accuracy")
    ax.plot(T, [r["macro_f1"] for r in curve], "s--", label="macro-F1")
    ax.axvline(t_train, color="gray", ls=":", label="T_train")
    ax.set_xlabel(f"{stage} loop count T (test time)")
    ax.set_ylabel("score")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, stem)


# ------------------------------------------------------------------ full report
def write_report(out_dir, probs, labels, classes, items=None, title="", extra=None):
    """Write every artefact for one evaluation. Returns the metrics dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    C = len(classes)
    pred = probs.argmax(1)
    m = compute_metrics(probs, labels, C)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({**(extra or {}), "metrics": m}, f, indent=2)

    cm = confusion_matrix(labels, pred, labels=list(range(C)))
    for name, data, fmt in (("confusion_matrix", cm, "{:d}"),
                            ("confusion_matrix_normalized", cm / cm.sum(1, keepdims=True).clip(min=1), "{:.4f}")):
        with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["true \\ pred"] + classes)
            for c, row in zip(classes, data):
                w.writerow([c] + [fmt.format(v) for v in row])
    plot_confusion(cm, classes, out / "confusion_matrix", False, title)
    plot_confusion(cm, classes, out / "confusion_matrix_normalized", True, title)

    p, r, f1, sup = precision_recall_fscore_support(labels, pred, labels=list(range(C)), zero_division=0)
    rows = [dict(class_id=i, **{"class": c}, precision=float(p[i]), recall=float(r[i]), f1=float(f1[i]),
                 support=int(sup[i]), errors=int(sup[i] - cm[i, i])) for i, c in enumerate(classes)]
    with open(out / "per_class.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for row in rows:
            w.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()})
    plot_per_class_f1(rows, out / "per_class_f1", title)
    with open(out / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(labels, pred, labels=list(range(C)), target_names=classes, digits=4,
                                      zero_division=0))
    plot_reliability(probs, labels, out / "reliability", title=title)

    with open(out / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred", "confidence", "correct"])
        for i in range(len(labels)):
            w.writerow([items[i][0] if items else i, classes[labels[i]], classes[pred[i]],
                        f"{probs[i, pred[i]]:.4f}", int(pred[i] == labels[i])])
    return m


# ------------------------------------------------------------------ efficiency
@torch.no_grad()
def throughput(model, img, device, amp_dtype=None, T=None, batch=64, iters=30, warmup=10):
    """Inference images/s at `batch` and single-image latency (ms). Leaves the model in eval mode."""
    model.eval()
    dev_type = torch.device(device).type
    if dev_type == "cpu":
        iters, warmup = 3, 1

    def run(bs, n):
        x = torch.randn(bs, 3, img, img, device=device)
        with torch.autocast(device_type=dev_type, dtype=amp_dtype, enabled=amp_dtype is not None):
            for _ in range(warmup):
                model(x, T=T)
            if dev_type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(n):
                model(x, T=T)
            if dev_type == "cuda":
                torch.cuda.synchronize()
        return time.perf_counter() - t

    ips = batch * iters / run(batch, iters)
    lat = 1000 * run(1, iters) / iters
    return {"throughput_img_s": round(ips, 1), "latency_ms_bs1": round(lat, 2), "batch": batch,
            "amp": str(amp_dtype).replace("torch.", "") if amp_dtype else "fp32",
            "gpu": torch.cuda.get_device_name() if dev_type == "cuda" else "cpu"}
