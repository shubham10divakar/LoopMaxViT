"""Splits, metrics and report artefacts. Run: python -m pytest tests -q"""
from collections import Counter

import numpy as np
import pytest

from loopmaxvit.data import stratified_split
from loopmaxvit.report import compute_metrics, write_report

C = 4


def pool(n_per_class=50):
    return [(f"img_{c}_{i}.jpg", c) for c in range(C) for i in range(n_per_class)]


def test_three_way_split_is_disjoint_stratified_and_seeded():
    items = pool()
    tr, va, te = stratified_split(items, 0.1, 0.1, seed=42)
    assert (len(tr), len(va), len(te)) == (160, 20, 20)
    assert not ({p for p, _ in tr} & {p for p, _ in va} or {p for p, _ in tr} & {p for p, _ in te}
                or {p for p, _ in va} & {p for p, _ in te})
    assert set(Counter(y for _, y in va).values()) == {5} and set(Counter(y for _, y in te).values()) == {5}
    assert stratified_split(items, 0.1, 0.1, seed=42) == (tr, va, te)


def test_two_way_split_has_no_test():
    tr, va, te = stratified_split(pool(), 0.2, 0.0, seed=42)
    assert (len(tr), len(va), te) == (160, 40, [])


def test_metrics_perfect_and_random():
    labels = np.repeat(np.arange(C), 10)
    m = compute_metrics(np.eye(C)[labels] * 0.97 + 0.01, labels, C)
    for k in ("acc", "balanced_acc", "macro_f1", "weighted_f1", "cohen_kappa", "mcc", "roc_auc_macro", "top5_acc"):
        assert m[k] == pytest.approx(1.0), k
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(C), len(labels))
    m = compute_metrics(p, labels, C)
    assert 0 <= m["acc"] <= 1 and m["top5_acc"] == 1.0 and m["ece"] >= 0


def test_write_report_artefacts(tmp_path):
    labels = np.repeat(np.arange(C), 10)
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(C), len(labels))
    classes = [f"Plant___disease_{i}" for i in range(C)]
    m = write_report(tmp_path, probs, labels, classes, [(f"x{i}.jpg", int(y)) for i, y in enumerate(labels)])
    assert m["n"] == len(labels)
    for f in ("metrics.json", "per_class.csv", "classification_report.txt", "predictions.csv",
              "confusion_matrix.csv", "confusion_matrix_normalized.csv", "confusion_matrix.png",
              "confusion_matrix_normalized.pdf", "per_class_f1.png", "reliability.png"):
        assert (tmp_path / f).exists(), f
