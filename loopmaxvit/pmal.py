"""
pmal.py: M1 "Permuted Multi-Axis Lattices" (design doc 02 sec. 4), pure NumPy.

Iteration t of a looped MaxViT block applies a fixed spatial permutation pi_t before partitioning and
pi_t^-1 after it, so iteration t attends over its own sparse graph G_t:

  window axis   pi^w_t = cyclic roll by s_t = (t * floor(P/2)) mod P in both axes          (sec. 4.1)
  grid axis     pi^g_t = block shear with (k_c, k_r) cycling (0,0) (1,0) (0,1) (1,1)       (sec. 4.2)
                    i1 = (i + k_r * (j // m)) mod H,   j1 = (j + k_c * (i1 // m)) mod W,   m = H / G
                (a roll or a plain row shear keeps the residue classes, i.e. the grid graph, unchanged)

A permutation `perm` is a gather index over the H*W positions: position k of the permuted map holds
the original position perm[k] (x_perm = x.flatten[perm]); inverse_perm(perm) undoes it.

Also here: group labels and graph overlap (the sec. 4.2 diagnostic), Swin-style masks for the PM3 ablation,
and union-graph rollout coverage (the sec. 4.5 mixing-time measure, PM5).
"""
import numpy as np

SHEAR_SCHEDULE = [(0, 0), (1, 0), (0, 1), (1, 1)]          # (k_c, k_r), repeated every 4 iterations


# ------------------------------------------------------------------ permutations
def identity_perm(h, w):
    return np.arange(h * w, dtype=np.int64)


def roll_perm(h, w, sh, sw):
    """Permutation equivalent to rolling the map by (-sh, -sw) (Swin-style cyclic shift)."""
    i, j = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return (((i + sh) % h) * w + (j + sw) % w).reshape(-1).astype(np.int64)


def block_shear_perm(h, w, m, k, kr=0):
    """Column shift by k * (block-row index i1 // m), after a row shift by kr * (block-column index j // m).
    Unlike a plain row shear this changes the residue classes that grid attention groups (m = H / G)."""
    i, j = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    i1 = (i + kr * (j // m)) % h
    j1 = (j + k * (i1 // m)) % w
    return (i1 * w + j1).reshape(-1).astype(np.int64)


def inverse_perm(p):
    inv = np.empty_like(p)
    inv[p] = np.arange(len(p), dtype=p.dtype)
    return inv


def window_shift(t, P):
    return (t * (P // 2)) % P


def window_perm(t, h, w, P):
    s = window_shift(t, P)
    return roll_perm(h, w, s, s)


def grid_perm(t, h, w, G):
    k, kr = SHEAR_SCHEDULE[t % len(SHEAR_SCHEDULE)]
    return block_shear_perm(h, w, h // G, k, kr)


def schedule(kind, t_max, h, w, size):
    """(t_max, H*W) gather indices for iterations 0 .. t_max-1 of one axis; row 0 is the identity."""
    fn = window_perm if kind == "window" else grid_perm
    return np.stack([fn(t, h, w, size) for t in range(t_max)])


# ------------------------------------------------------------------ attention graphs
def group_of(kind, h, w, size):
    """Group id of every position (row-major) under window_partition / grid_partition with this size."""
    i, j = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    if kind == "window":
        g = (i // size) * (w // size) + j // size
    else:
        mh, mw = h // size, w // size
        g = (i % mh) * mw + j % mw
    return g.reshape(-1)


def group_labels(perm, kind, h, w, size):
    """Group of every ORIGINAL token when the map is permuted by `perm` before partitioning."""
    lab = np.empty(h * w, dtype=np.int64)
    lab[perm] = group_of(kind, h, w, size)
    return lab


def group_overlap(perm_a, perm_b, kind, h, w, size):
    """Fraction of same-group token pairs under perm_a that are also in one group under perm_b
    (1.0 = identical graphs). Pairs include i == j, as in the design's reference diagnostic."""
    la, lb = group_labels(perm_a, kind, h, w, size), group_labels(perm_b, kind, h, w, size)
    same_a = la[:, None] == la[None, :]
    same_b = lb[:, None] == lb[None, :]
    return float((same_a & same_b).sum() / same_a.sum())


def n_distinct_graphs(perms, kind, h, w, size):
    """How many different partitions (as sets of groups) a list of permutations produces."""
    seen = set()
    for p in perms:
        lab = group_labels(p, kind, h, w, size)
        seen.add(frozenset(frozenset(np.flatnonzero(lab == g).tolist()) for g in np.unique(lab)))
    return len(seen)


# ------------------------------------------------------------------ PM3: Swin-style mask for cyclic windows
def window_mask(h, w, P, shift, neg=-100.0):
    """(nW, P*P, P*P) additive mask for the window partition of a map rolled by `shift`: tokens that only share
    a window because of the cyclic wrap-around may not attend to each other (Swin). Zeros when shift = 0."""
    i, j = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    region = ((i >= h - shift) * 2 + (j >= w - shift)) if shift else np.zeros((h, w), np.int64)
    win = region.reshape(h // P, P, w // P, P).transpose(0, 2, 1, 3).reshape(-1, P * P)
    return np.where(win[:, :, None] == win[:, None, :], 0.0, neg).astype(np.float32)


# ------------------------------------------------------------------ PM5: mixing time
def uniform_attention_matrix(perm, kind, h, w, size):
    """N x N row-stochastic matrix of uniform attention inside each group (structure only, no weights)."""
    lab = group_labels(perm, kind, h, w, size)
    same = (lab[:, None] == lab[None, :]).astype(np.float64)
    return same / same.sum(1, keepdims=True)


def rollout_coverage(mats, centre, frac=0.5, residual=0.5):
    """Attention rollout (Abnar & Zuidema 2020) through a sequence of N x N attention matrices, each mixed with
    the residual path: M = residual * I + (1 - residual) * A.  Returns, after each matrix, the fraction of
    positions whose influence from `centre` is >= frac / N (frac = 1 would be a perfectly uniform share)."""
    n = mats[0].shape[0]
    infl = np.zeros(n)
    infl[centre] = 1.0
    cov = []
    for A in mats:
        infl = residual * infl + (1 - residual) * (A @ infl)       # column `centre` of the rollout product
        cov.append(float((infl >= frac / n).mean()))
    return cov


def mixing_time(coverage_per_iter, target=0.9):
    """First iteration (1-based) at which coverage >= target, or None."""
    for t, c in enumerate(coverage_per_iter, 1):
        if c >= target:
            return t
    return None


def structural_coverage(h, w, P, G, T, permuted=True, axes=("window", "grid"), frac=0.5, centre=None):
    """Coverage after each of T iterations of a looped (window -> grid) block with uniform attention.
    permuted=False is the fixed-partition loop (Loop-plain)."""
    centre = (h // 2) * w + w // 2 if centre is None else centre
    ident = identity_perm(h, w)
    mats, per_iter = [], []
    for t in range(T):
        pw = window_perm(t, h, w, P) if permuted and "window" in axes else ident
        pg = grid_perm(t, h, w, G) if permuted and "grid" in axes else ident
        mats += [uniform_attention_matrix(pw, "window", h, w, P), uniform_attention_matrix(pg, "grid", h, w, G)]
    cov = rollout_coverage(mats, centre, frac)
    return cov[1::2]                                                 # after each full block (both axes)
