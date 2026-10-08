# LoopMaxViT (PyTorch)

Implementation of design doc **02 — LoopMaxViT: looped multi-axis attention with permuted lattices** (and the
shared rules in doc 00) from `looped-hybrids/looped-hybrids/`. Every MaxViT stage becomes a unique stride-2
transition block plus **one shared MaxViT block (MBConv → block attention → grid attention) applied T times**, with
input injection, iteration embeddings, per-iteration step sizes, per-iteration BatchNorm, random-T training and
BN recalibration.

The primary mechanism is **M1 PMAL (Permuted Multi-Axis Lattices)**. Iteration t applies a fixed spatial
permutation π_t before partitioning and π_t⁻¹ after it, so the T passes of one block attend over T *different*
sparse graphs:

- **Window axis**: a cyclic roll by s_t = (t·⌊P/2⌋) mod P (P = 7 gives shifts 0, 3, 6, 2, 5, 1, 4).
- **Grid axis**: a *block shear* (k_c, k_r) cycling (0,0) → (1,0) → (0,1) → (1,1). A roll or a plain row shear
  keeps the residue classes, so the grid graph would not change at all.
- **Per-iteration relative-bias tables**, because the permutation changes the true offsets inside a group.

The design's reference code is Keras 3 + TensorFlow. This repo is a **PyTorch** port, because TensorFlow has no
native-Windows GPU support. It has the same training, evaluation and reporting pipeline as the LoopCoAtNet repo.

## See the architecture

```bash
python show_arch.py                                    # maxvit-t, Loop-full (PMAL), 224 px
python show_arch.py --graphs                           # PMAL graph overlaps, distinct graphs, mixing time per stage
python show_arch.py --compare                          # every mode side by side (params, GMACs)
python show_arch.py --variant maxvit-b --mode plain --carry
python show_arch.py --dataset paddy --summary --depth 4
python show_arch.py --checkpoint runs/<run>/best.pt
```

## Models (`--variant`, `--mode`, flags)

| mode / flag | looped stages | role (doc 00 sec. 8.2 / doc 02 sec. 5, 7) |
|---|---|---|
| `base` | L−1 distinct MaxViT blocks per stage | Base MaxViT |
| `shallow` | 1 distinct block per stage | Shallow-matched |
| `plain` | 1 shared MaxViT block × T, **fixed** partitions | Loop-plain (PM1 control) |
| `pmal` (default) | 1 shared MaxViT block × T with **M1 PMAL** | Loop-full |
| `--carry` | S3 attention logits carried across iterations, α per head **per axis** | M3 |
| `--pmal-axes window\|grid` | permute one axis only | ablation PM2 |
| `--window-mask` | Swin-masked instead of cyclic shifted windows | ablation PM3 |
| `--rel-tables shared\|per_iter` | 1 table vs t_max tables | ablation PM4 |
| `model.s3_cycle=k` | k distinct shared blocks cycled in S3 | R3c |

Variants: `maxvit-nano` (dev), `maxvit-t`, `maxvit-s`, `maxvit-b` (paper headline, −56 %). Default loop counts
(T = L−1): MaxViT-T S1/S2/S3/S4 = 1/1/**4**/1 and MaxViT-B = 1/5/**13**/1. Random T per batch follows
U[⌈0.5T⌉, ⌈1.5T⌉] (S3 of T: U[2,6]). S1 and S4 use U[1,2] and U[1,3]. Truncated BPTT (k = 4) turns on when T can
exceed 8 (MaxViT-B). `model.loop_stages` selects which stages loop.

**Resolution:** `P = G = img/32` by default (224→7, 192→6, 160→5, 128→4, 64→2), which divides every stage map.
For the doc's richer-lattice check, use `--img 320 --P 5` (S3 = 20², m = 4).

## PMAL diagnostics and PM5 mixing time

`show_arch.py --graphs` prints these numbers. They are also saved in every run's `model_summary.txt`. They
reproduce the doc's §4.2 table exactly, for S3 at 224 px:

| iteration t | window graph vs t=0 | grid graph vs t=0 |
|---|---|---|
| 1 | 0.26 | 0.51 |
| 2 | 0.57 | 0.51 |
| 3 | 0.35 | 0.26 |

Under uniform-attention rollout from the centre pixel, PMAL reaches 90 % coverage of S3 in **2** iterations. Fixed
partitions need 3. At S4 (7×7 = one window), PMAL changes no graph, only the per-iteration bias.

`mixing_time.py` measures the same quantity with the **learned** attention of trained checkpoints (attention
rollout, doc §4.5, experiment PM5):

```bash
python mixing_time.py runs/<pmal run>/best.pt runs/<plain run>/best.pt   # -> report/mixing_time.{json,png,pdf}
python mixing_time.py --variant maxvit-t                                 # untrained models + structure only
```

## Datasets and splits

These work the same as in LoopCoAtNet. The registry is `configs/datasets.yaml` (`paddy`, `rice-leaf-bd`,
`plantvillage`, `plantwild`), and images are read in place from the shared Datasets folder. Override the location
with `--data-root` or `LOOPMAXVIT_DATA_ROOT`. Splits are stratified with seed 42 and saved to `split.json`:
`train-val-test` 80/10/10 (default) or `train-val` 80/20. `--official-split` uses PlantWild's official split.

## Train / evaluate

```bash
python train.py                                         # paddy, maxvit-t PMAL, 80/10/10
python train.py --mode plain                            # PM1: fixed-partition loop
python train.py --mode base                             # baseline
python train.py --pmal-axes grid                        # PM2
python train.py --window-mask                           # PM3
python train.py --rel-tables shared                     # PM4
python train.py --carry                                 # + M3
python train.py --variant maxvit-b --set model.drop_path=0.3
python train.py --set train.epochs=5 train.limit=512   # quick sanity run
python train.py --resume runs/<run>/last.pt             # rerunning the same command also resumes
python evaluate.py runs/<run>/best.pt --curve           # PM6: acc vs test-time T (1 .. 2 x T_train)
python evaluate.py runs/<run>/best.pt --T stage3=8
python aggregate.py runs/paddy_*                        # paper table: mean ± std over seeds (.csv/.md/.tex)
python -m pytest                                        # 37 unit tests
```

Training features (shared with LoopCoAtNet):
- **Run header and `model_summary.txt`**: stage plan, PMAL graphs, torchinfo table and module tree.
- **Early stopping** on `val_acc`, `val_macro_f1` or `val_loss`.
- **Atomic checkpoints**: `last.pt`, `best.pt` and rotating `checkpoints/epoch_XXX.pt`, with exact resume.
- **Final report** for best and last checkpoints on test and val: accuracy, balanced accuracy, top-5,
  macro/weighted P/R/F1, κ, MCC, ROC-AUC, ECE and NLL, plus per-class CSVs and plots, confusion matrices,
  reliability diagrams and predictions.
- **Plots and efficiency**: training curves, the acc-vs-T curve, fixed-point residuals, learned
  γ/β/e/α diagnostics, params, GMACs, throughput and latency.

**Recipe** (doc 00 sec. 7):
- AdamW with lr 1e-3·batch/512 and weight decay 0.05 (none on norms, biases, loop parameters, α or relative
  tables).
- Warmup, then cosine.
- Grad clip 1.0.
- RandAugment(2,9), Mixup/CutMix and RandomErasing.
- Label smoothing 0.1.
- bf16 autocast.
- Drop path 0.1, applied per iteration.
- BN recalibration before every evaluation (R13).

The default batch is **32**. MaxViT-T at 224 px with the largest random T peaks at 8.1 GB. Batch 64 needs
~16 GB and spills into shared memory on a 12 GB card.

## Parity with the design doc (224 px, 1000 classes)

| model | doc 02 (Keras) | this repo |
|---|---|---|
| MaxViT-T base / PMAL | 29.12M / 22.48M | 29.12M / 22.50M |
| MaxViT-B base / PMAL | 115.46M / 50.28M | 115.46M / 50.31M |
| nano base / PMAL (64 px, P = 2) | 7.21M / 5.55M | 7.21M / 5.56M |

Like Keras' `count_params()`, these counts include BN running statistics. They were measured with the doc's
setting, where S2–S4 are looped. The extra 0.02M comes from S4, which allocates t_max = 3 so that random
T ~ U[1,3] has its own parameters. The default here also loops S1 (doc §3), which gives 22.42M trainable for
MaxViT-T.

## Not implemented

- **M6 halting**: optional in the doc.
- **Fused inference permutations**: π_t is applied as one `index_select` per attention. It is not folded into the
  partition reshape.

## Layout

```
loopmaxvit/pmal.py          M1 permutations (roll, block shear), group overlap, Swin mask (PM3), rollout coverage
loopmaxvit/blocks.py        MBConv, partitions, RelPosMHSA (per-iteration tables), AxisAttention, MaxViTBlock
loopmaxvit/model.py         LoopMaxViT, variants, modes and ablation flags
loopmaxvit/looped_core.py   IterNorm, LoopedStage, UnrolledStage, CycleBlock, sample_T, recalibrate_bn
loopmaxvit/data.py engine.py report.py summary.py      data, training utilities, metrics/plots, architecture printout
show_arch.py  train.py  evaluate.py  aggregate.py  mixing_time.py  tests/  configs/{default,datasets}.yaml
```
