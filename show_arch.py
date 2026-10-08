"""
show_arch.py: print the LoopMaxViT architecture.

  python show_arch.py                                   # maxvit-t, Loop-full (PMAL, M1), 224 px, 1000 classes
  python show_arch.py --dataset paddy                   # num_classes taken from the dataset (10)
  python show_arch.py --variant maxvit-b --mode plain --carry
  python show_arch.py --graphs                          # PMAL graph overlaps, distinct graphs, mixing time
  python show_arch.py --compare                         # params / GMACs of every mode side by side
  python show_arch.py --summary --depth 4               # + torchinfo layer table
  python show_arch.py --tree                            # + full PyTorch module tree
  python show_arch.py --checkpoint runs/<run>/best.pt   # the architecture of a trained run
"""
import argparse
import sys

import torch

from loopmaxvit.model import CFG, MODES, PMAL_AXES, REL_TABLES, build_model
from loopmaxvit.summary import gmacs, print_compare, print_graphs, print_loops, print_plan


def parse_T(items):
    T = {}
    for it in items or []:
        k, v = it.split("=")
        T[k if k.startswith("stage") else "stage" + k] = int(v)
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="maxvit-t", choices=list(CFG))
    ap.add_argument("--mode", default="pmal", choices=MODES)
    ap.add_argument("--carry", action="store_true", help="M3 per-axis loop-carried attention logits in S3")
    ap.add_argument("--pmal-axes", default="both", choices=list(PMAL_AXES))
    ap.add_argument("--window-mask", action="store_true")
    ap.add_argument("--rel-tables", default="auto", choices=REL_TABLES)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--P", default="auto", help="window = grid size (auto = img/32)")
    ap.add_argument("--num-classes", type=int, default=1000)
    ap.add_argument("--dataset", help="take num_classes from a registered dataset (configs/datasets.yaml)")
    ap.add_argument("--data-root")
    ap.add_argument("--T", nargs="*", help="training T per stage, e.g. --T stage3=4 stage2=2")
    ap.add_argument("--checkpoint", help="show the architecture stored in a training checkpoint")
    ap.add_argument("--compare", action="store_true", help="table of every mode (params, GMACs)")
    ap.add_argument("--graphs", action="store_true", help="PMAL attention-graph diagnostics per looped stage")
    ap.add_argument("--summary", action="store_true", help="torchinfo layer-by-layer table")
    ap.add_argument("--depth", type=int, default=3, help="torchinfo depth")
    ap.add_argument("--tree", action="store_true", help="print the full PyTorch module tree")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")    # torchinfo draws box characters (cp1252 on Windows)

    kw = dict(variant=a.variant, mode=a.mode, img=a.img, num_classes=a.num_classes, carry=a.carry, T=parse_T(a.T),
              pmal_axes=a.pmal_axes, window_mask=a.window_mask, rel_tables=a.rel_tables, P=a.P)
    if a.checkpoint:
        ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        kw = {**ck["model_cfg"], "num_classes": len(ck["classes"])}
        print(f"checkpoint: {a.checkpoint}  (dataset {ck.get('dataset')}, epoch {ck.get('epoch')})")
    elif a.dataset:
        from loopmaxvit.data import load_registry, build_splits
        classes = build_splits(a.dataset, load_registry(), a.data_root)[0]
        kw["num_classes"] = len(classes)
        print(f"dataset {a.dataset}: {len(classes)} classes")

    model = build_model(**kw).eval()
    if a.checkpoint:
        model.load_state_dict(ck["model"])
    print_plan(model, kw["img"])
    print_loops(model)
    print(f" GMACs at nominal T {model.train_T() or '-'}: {gmacs(model, kw['img']):.2f}   "
          f"(looping cuts parameters, not compute: FLOPs ~ T x block FLOPs)\n")
    if a.graphs:
        print_graphs(model)
    if a.compare:
        print_compare(kw["variant"], kw["img"], kw["num_classes"])
    if a.summary:
        from torchinfo import summary
        print(summary(model, input_size=(1, 3, kw["img"], kw["img"]), depth=a.depth, verbose=0,
                      col_names=("output_size", "num_params"), row_settings=("var_names",)))
    if a.tree:
        print(model)


if __name__ == "__main__":
    main()
