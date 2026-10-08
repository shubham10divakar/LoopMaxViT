"""M1 PMAL permutations and graph diagnostics (design doc 02 sec. 4). Run: python -m pytest tests -q"""
import numpy as np
import pytest
import torch

from loopmaxvit import pmal
from loopmaxvit.blocks import grid_partition, spatial_permute, window_partition


@pytest.mark.parametrize("h,size", [(14, 7), (28, 7), (56, 7), (16, 4), (8, 2)])
def test_schedules_are_bijections_with_identity_first(h, size):
    for kind in ("window", "grid"):
        perms = pmal.schedule(kind, 8, h, h, size)
        assert np.array_equal(perms[0], np.arange(h * h))
        for p in perms:
            assert np.array_equal(np.sort(p), np.arange(h * h))
            assert np.array_equal(p[pmal.inverse_perm(p)], np.arange(h * h))


def test_window_shifts_cover_all_phases():
    assert [pmal.window_shift(t, 7) for t in range(7)] == [0, 3, 6, 2, 5, 1, 4]     # doc 02 sec. 4.1


def test_overlaps_match_design_doc():
    """Doc 02 sec. 4.2 table (S3 geometry 14x14, P = G = 7)."""
    h = w = 14
    ident = pmal.identity_perm(h, w)
    expected = {1: (0.26, 0.51), 2: (0.57, 0.51), 3: (0.35, 0.26)}
    for t, (ow, og) in expected.items():
        assert round(pmal.group_overlap(ident, pmal.window_perm(t, h, w, 7), "window", h, w, 7), 2) == ow
        assert round(pmal.group_overlap(ident, pmal.grid_perm(t, h, w, 7), "grid", h, w, 7), 2) == og
    assert pmal.n_distinct_graphs([pmal.grid_perm(t, h, w, 7) for t in range(4)], "grid", h, w, 7) == 4


def test_roll_and_row_shear_do_not_change_grid_graph():
    """Why PMAL needs a BLOCK shear: rolling keeps the residue classes (overlap 1.00, doc 02 sec. 4.2)."""
    h = w = 14
    ident = pmal.identity_perm(h, w)
    assert pmal.group_overlap(ident, pmal.roll_perm(h, w, 3, 3), "grid", h, w, 7) == 1.0


def test_group_labels_match_torch_partitions():
    """The NumPy group model agrees with the actual window_partition / grid_partition + spatial_permute."""
    h = w = 28
    for kind, part, size in (("window", window_partition, 7), ("grid", grid_partition, 7)):
        for t in range(4):
            p = (pmal.window_perm if kind == "window" else pmal.grid_perm)(t, h, w, size)
            pos = torch.arange(h * w).float().view(1, h, w, 1)
            groups = part(spatial_permute(pos, torch.from_numpy(p)), size)[..., 0].long()
            lab = pmal.group_labels(p, kind, h, w, size)
            for g in groups:
                assert len(set(lab[g.numpy()].tolist())) == 1


def test_window_mask_blocks_wraparound_pairs():
    m = pmal.window_mask(14, 14, 7, 3)
    assert m.shape == (4, 49, 49)
    assert (m[0] == 0).all()                         # top-left window does not wrap
    assert (m[3] < 0).any()                          # bottom-right window mixes wrapped regions
    assert (pmal.window_mask(14, 14, 7, 0) == 0).all()


def test_pmal_mixes_faster_than_fixed_partitions():
    """Sec. 4.5: the union graph of PMAL reaches 90 % coverage in fewer iterations (S3, 224 px)."""
    fixed = pmal.mixing_time(pmal.structural_coverage(14, 14, 7, 7, 6, permuted=False))
    perm = pmal.mixing_time(pmal.structural_coverage(14, 14, 7, 7, 6, permuted=True))
    assert perm < fixed
