"""
aggregate.py: collect runs/*/final.json into a paper table, mean ± std over seeds.

  python aggregate.py                                   # every run under runs/
  python aggregate.py runs/plantvillage_*               # selected runs
  python aggregate.py --out results/plantvillage        # -> .csv, .md, .tex

Rows are grouped by (dataset, model, split); the metrics are the headline numbers (best.pt on test, or on
val for train-val runs, which are marked).
"""
import argparse
import csv
import glob
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
COLS = [("acc", "Acc", 100), ("balanced_acc", "Bal. Acc", 100), ("precision_macro", "Prec.", 100),
        ("recall_macro", "Rec.", 100), ("macro_f1", "Macro-F1", 100), ("weighted_f1", "W-F1", 100),
        ("cohen_kappa", "Kappa", 100), ("mcc", "MCC", 100), ("roc_auc_macro", "AUC", 100),
        ("top5_acc", "Top-5", 100), ("ece", "ECE", 100)]


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run directories (default: runs/*)")
    ap.add_argument("--out", default=str(REPO / "results" / "results"), help="output path stem")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    dirs = [Path(p) for r in (a.runs or [str(REPO / "runs" / "*")]) for p in glob.glob(r)]
    finals = []
    for d in dirs:
        f = d / "final.json"
        if f.exists():
            j = json.load(open(f, encoding="utf-8"))
            if "headline" in j:                  # skip runs from before the reporting update
                finals.append(j)
    if not finals:
        raise SystemExit("no final.json with a headline found")

    groups = {}
    for j in finals:
        groups.setdefault((j["dataset"], j["model"], j.get("split", "train-val")), []).append(j)

    rows = []
    for (ds, model, split), js in sorted(groups.items()):
        row = {"dataset": ds, "model": model, "split": split, "eval_on": js[0]["eval_split"],
               "seeds": ",".join(str(j["seed"]) for j in js), "n": len(js),
               "params_M": js[0]["efficiency"]["params_M"], "gmacs": js[0]["efficiency"]["gmacs"],
               "throughput": js[0]["efficiency"].get("throughput_img_s"),
               "epochs": mean_std([j["epochs_run"] for j in js])[0]}
        for k, _, scale in COLS:
            m, s = mean_std([j["headline"].get(k) for j in js])
            row[k] = None if m is None else (m * scale, s * scale)
        rows.append(row)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pm = lambda v, d=2: "-" if v is None else f"{v[0]:.{d}f} ± {v[1]:.{d}f}"   # noqa: E731

    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "model", "split", "eval_on", "n_seeds", "params_M", "GMACs", "img_s"]
                   + [x for k, _, _ in COLS for x in (k + "_mean", k + "_std")])
        for r in rows:
            w.writerow([r["dataset"], r["model"], r["split"], r["eval_on"], r["n"], r["params_M"], r["gmacs"],
                        r["throughput"]] + [x for k, _, _ in COLS for x in
                                            ((f"{r[k][0]:.4f}", f"{r[k][1]:.4f}") if r[k] else ("", ""))])

    head = ["Dataset", "Model", "Eval", "Seeds", "Params (M)", "GMACs"] + [c for _, c, _ in COLS]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        ev = r["eval_on"] + ("*" if r["eval_on"] == "val" else "")
        md.append("| " + " | ".join([r["dataset"], r["model"], ev, str(r["n"]), f"{r['params_M']:.2f}",
                                     f"{r['gmacs']:.2f}"] + [pm(r[k]) for k, _, _ in COLS]) + " |")
    md.append("\nValues in %, mean ± std over seeds. * = train-val run, evaluated on the val split that also "
              "selected the checkpoint (optimistic).")
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")

    tex_cols = ["acc", "macro_f1", "cohen_kappa", "mcc", "ece"]
    names = {k: c for k, c, _ in COLS}
    tex = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\begin{tabular}{ll" + "r" * (2 + len(tex_cols)) + "}", r"\toprule",
           "Dataset & Model & Params (M) & GMACs & " + " & ".join(names[k] for k in tex_cols) + r" \\",
           r"\midrule"]
    for r in rows:
        cells = ["-" if r[k] is None else f"{r[k][0]:.2f} $\\pm$ {r[k][1]:.2f}" for k in tex_cols]
        tex.append(f"{r['dataset']} & {r['model'].replace('_', chr(92) + '_')} & {r['params_M']:.2f} & "
                   f"{r['gmacs']:.2f} & " + " & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Test-set results (\%, mean $\pm$ std over seeds).}", r"\end{table}"]
    out.with_suffix(".tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    print("\n".join(md))
    print(f"\nsaved {out.with_suffix('.csv')}, .md, .tex")


if __name__ == "__main__":
    main()
