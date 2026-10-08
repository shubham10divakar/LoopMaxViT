"""
data.py: dataset registry, stratified splits and loaders. Images are read in place from the
shared Datasets folder (configs/datasets.yaml); nothing is copied.
"""
import json
import os
from pathlib import Path

import torch
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as Tv

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REPO = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = REPO / "configs" / "datasets.yaml"


def load_registry(path=DEFAULT_REGISTRY):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_root(registry, cli_root=None):
    return Path(cli_root or os.environ.get("LOOPMAXVIT_DATA_ROOT") or registry["data_root"])


# ------------------------------------------------------------------ scanners
def scan_folder(root):
    """root/<class>/<image>; non-directory entries (e.g. a stray .zip) are ignored."""
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    items = []
    for ci, c in enumerate(classes):
        for p in sorted((root / c).iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                items.append((str(p), ci))
    return classes, items


def scan_fiftyone(root, entry):
    """FiftyOne export (PlantWild): labels in samples.json. Returns classes, items, per-item split."""
    with open(root / "samples.json", "r", encoding="utf-8") as f:
        samples = json.load(f)["samples"]
    flt = entry.get("filter", {})
    samples = [s for s in samples if all(s.get(k) == v for k, v in flt.items())]
    lf, sf = entry.get("label_field", "ground_truth"), entry.get("split_field", "split")
    classes = sorted({s[lf]["label"] for s in samples})
    cidx = {c: i for i, c in enumerate(classes)}
    items = [(str(root / s["filepath"]), cidx[s[lf]["label"]]) for s in samples]
    splits = [s.get(sf) for s in samples]
    return classes, items, splits


SPLITS = ("train-val-test", "train-val")


def default_val_frac(split):
    """80/10/10 for train-val-test, 80/20 for train-val."""
    return 0.1 if split == "train-val-test" else 0.2


def stratified_split(items, val_frac, test_frac, seed):
    """Stratified (train, val, test) from one labelled pool; fractions are of the whole pool.
    test_frac = 0 gives an empty test list."""
    labels = [y for _, y in items]
    test = []
    if test_frac > 0:
        items, test = train_test_split(items, test_size=test_frac, stratify=labels, random_state=seed)
        labels = [y for _, y in items]
    train, val = train_test_split(items, test_size=val_frac / (1 - test_frac), stratify=labels, random_state=seed)
    return train, val, test


def build_splits(name, registry, data_root=None, split="train-val-test", val_frac=None, test_frac=0.1, seed=42,
                 official_split=False):
    """Returns (classes, train_items, val_items, test_items). items = [(path, label)].

    split = "train-val-test": train / val (model selection, early stopping) / test (reported once).
    split = "train-val":      train / val only, test_items = []; numbers reported on val are optimistic,
                              because val also picks the best checkpoint.
    official_split (plantwild): train-val-test -> official train / val / test;
                                train-val      -> official train+val / test (the paper protocol)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    val_frac = default_val_frac(split) if val_frac is None else val_frac
    test_frac = test_frac if split == "train-val-test" else 0.0
    entry = registry["datasets"][name]
    root = resolve_root(registry, data_root) / entry["path"]
    if not root.exists():
        raise FileNotFoundError(f"dataset '{name}' not found at {root} (set --data-root or LOOPMAXVIT_DATA_ROOT)")
    if entry["format"] == "folder":
        if official_split:
            raise ValueError(f"dataset '{name}' has no official split")
        classes, items = scan_folder(root)
    elif entry["format"] == "fiftyone":
        classes, items, splits = scan_fiftyone(root, entry)
        if official_split:
            off = entry["official"]
            pick = lambda keys: [it for it, s in zip(items, splits) if s in keys]   # noqa: E731
            if split == "train-val-test":
                return classes, pick(off["train"]), pick(off["val"]), pick(off["test"])
            return classes, pick(off["train"] + off["val"]), pick(off["test"]), []
    else:
        raise ValueError(f"unknown format {entry['format']}")
    return (classes, *stratified_split(items, val_frac, test_frac, seed))


# ------------------------------------------------------------------ dataset
class ImageList(Dataset):
    def __init__(self, items, transform, draft_size=None):
        self.items, self.transform, self.draft_size = items, transform, draft_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        with Image.open(path) as im:
            if self.draft_size:                  # JPEG: decode at reduced scale (big speed-up on 1024px+ images)
                im.draft("RGB", (self.draft_size, self.draft_size))
            im = im.convert("RGB")
        return self.transform(im), y


def make_transforms(img, mean, std, aug):
    """Doc 00 sec. 7: RandAugment(2, 9), RandomErasing 0.25 (Mixup/CutMix are applied per batch in train.py)."""
    train = [Tv.RandomResizedCrop(img, scale=tuple(aug.get("crop_scale", (0.25, 1.0)))),
             Tv.RandomHorizontalFlip()]
    if aug.get("vflip"):
        train.append(Tv.RandomVerticalFlip())
    if aug.get("randaugment"):
        n, m = aug["randaugment"]
        train.append(Tv.RandAugment(num_ops=n, magnitude=m))
    train += [Tv.ToTensor(), Tv.Normalize(mean, std)]
    if aug.get("random_erasing", 0) > 0:
        train.append(Tv.RandomErasing(p=aug["random_erasing"]))
    resize = int(round(img / aug.get("eval_crop_pct", 0.875)))
    evalt = Tv.Compose([Tv.Resize(resize), Tv.CenterCrop(img), Tv.ToTensor(), Tv.Normalize(mean, std)])
    return Tv.Compose(train), evalt


def build_loaders(name, img, batch_size, aug, registry=None, data_root=None, split="train-val-test", val_frac=None,
                  test_frac=0.1, seed=42, official_split=False, num_workers=8, limit=None):
    """Returns dict(train, val, test, recal, classes, train_items, val_items, test_items, mean, std).
    test is None for split="train-val". recal = training images with eval transforms (BN recalibration, R13)."""
    registry = registry or load_registry()
    entry = registry["datasets"][name]
    classes, tr, va, te = build_splits(name, registry, data_root, split, val_frac, test_frac, seed, official_split)
    if limit:                                        # quick sanity runs: deterministic subsample
        g = torch.Generator().manual_seed(seed)
        sub = lambda xs, n: [xs[i] for i in torch.randperm(len(xs), generator=g)[:n].tolist()]   # noqa: E731
        tr, va, te = sub(tr, limit), sub(va, max(1, limit // 4)), sub(te, max(1, limit // 4)) if te else []
    mean, std = entry.get("mean", [0.485, 0.456, 0.406]), entry.get("std", [0.229, 0.224, 0.225])
    t_train, t_eval = make_transforms(img, mean, std, aug)
    draft = 2 * img
    kw = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    return dict(
        train=DataLoader(ImageList(tr, t_train, draft), batch_size, shuffle=True, drop_last=True, **kw),
        val=DataLoader(ImageList(va, t_eval, draft), batch_size, shuffle=False, **kw),
        test=DataLoader(ImageList(te, t_eval, draft), batch_size, shuffle=False, **kw) if te else None,
        recal=DataLoader(ImageList(tr, t_eval, draft), batch_size, shuffle=True, drop_last=True, **kw),
        classes=classes, train_items=tr, val_items=va, test_items=te, mean=mean, std=std)
