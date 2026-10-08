"""Unit tests (design doc 00 sec. 6 / doc 02 sec. 9). Run: python -m pytest tests -q"""
import random

import pytest
import torch
import torch.nn as nn

from loopmaxvit import LoopedStage, MaxViTBlock, MBConv, build_model, recalibrate_bn, sample_T
from loopmaxvit.model import MODES

IMG, NCLS = 64, 4          # dev geometry: stage maps 16, 8, 4, 2 -> P = G = 2


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def test_T1_equivalence_mbconv_and_maxvit_block():
    """With fresh init and T=1, LoopedStage(B)(x) == B(x) exactly."""
    torch.manual_seed(0)
    blk = MBConv(16, 16, t_max=4).eval()
    st = LoopedStage(blk, 16, t_train=1, t_max=4).eval()
    x = torch.randn(2, 16, 8, 8)
    assert torch.allclose(st(x), blk(x), atol=1e-6)
    blk = MaxViTBlock(32, 32, (8, 8), 4, 4, t_max=4, pmal_axes=("window", "grid")).eval()
    st = LoopedStage(blk, 32, t_train=1, t_max=4).eval()
    x = torch.randn(2, 32, 8, 8)
    assert torch.allclose(st(x), blk(x), atol=1e-5)


def test_pmal_t0_is_canonical_and_t1_differs():
    """Iteration 0 of a PMAL block is the plain MaxViT block; later iterations attend over other graphs."""
    torch.manual_seed(0)
    plain = MaxViTBlock(32, 32, (8, 8), 4, 4, t_max=4, rel_tables="per_iter").eval()
    perm = MaxViTBlock(32, 32, (8, 8), 4, 4, t_max=4, pmal_axes=("window", "grid")).eval()
    perm.load_state_dict(plain.state_dict())
    x = torch.randn(2, 32, 8, 8)
    assert torch.allclose(plain(x, t=0), perm(x, t=0), atol=1e-6)
    assert not torch.allclose(plain(x, t=1), perm(x, t=1), atol=1e-3)


def test_permuted_axis_roundtrip():
    """The stored inverse undoes pi_t, so a permuted axis returns a map in the canonical layout."""
    torch.manual_seed(0)
    blk = MaxViTBlock(32, 32, (8, 8), 4, 4, t_max=4, pmal_axes=("window", "grid")).eval()
    ax = blk.grid_att
    x = torch.randn(2, 8, 8, 32)
    y, _ = ax(x, t=1)
    assert y.shape == x.shape and torch.isfinite(y).all()
    from loopmaxvit.blocks import spatial_permute
    p, inv = ax.perm[1], ax.inv[1]
    assert torch.equal(spatial_permute(spatial_permute(x, p), inv), x)


@pytest.mark.parametrize("kw", [dict(mode=m) for m in MODES] + [
    dict(mode="plain", carry=True), dict(mode="pmal", carry=True), dict(mode="pmal", pmal_axes="window"),
    dict(mode="pmal", pmal_axes="grid"), dict(mode="pmal", window_mask=True), dict(mode="pmal", rel_tables="shared"),
    dict(mode="plain", rel_tables="per_iter"), dict(mode="pmal", s3_cycle=2), dict(mode="pmal", carry=True, s3_cycle=2)])
def test_forward_all_modes(kw):
    m = build_model("maxvit-nano", img=IMG, num_classes=NCLS, **kw).eval()
    x = torch.randn(2, 3, IMG, IMG)
    n = n_params(m)
    for T in ({}, {"stage3": 1}, {"stage3": 12, "stage2": 3, "stage4": 3}):   # incl. T > t_max (extrapolation)
        y = m(x, T=T)
        assert y.shape == (2, NCLS) and torch.isfinite(y).all()
        assert n_params(m) == n                                              # params independent of T
    m.train()
    assert m(x).shape == (2, NCLS)


def test_resolution_compatibility():
    for img, P in ((224, 7), (192, 6), (160, 5), (128, 4)):
        assert build_model("maxvit-nano", img=img, num_classes=2).P == P
    assert build_model("maxvit-nano", img=320, num_classes=2, P=5).P == 5      # doc 02 risk table: m = 4 at S3
    with pytest.raises(AssertionError):
        build_model("maxvit-nano", img=224, num_classes=2, P=5)


def test_param_parity_with_design_doc():
    """Doc 02: MaxViT-T 29.12M, MaxViT-B 115.46M (Keras counts include BN running statistics)."""
    keras = lambda m: (n_params(m) + sum(b.numel() for k, b in m.named_buffers() if "running" in k)) / 1e6  # noqa
    assert round(keras(build_model("maxvit-t", "base", 224, 1000)), 2) == 29.12
    assert round(keras(build_model("maxvit-b", "base", 224, 1000)), 2) == 115.46
    loop = keras(build_model("maxvit-b", "pmal", 224, 1000, loop_stages=("stage2", "stage3", "stage4")))
    assert abs(loop - 50.28) < 0.05                                           # -56 %


def test_gradients_reach_shared_and_loop_params():
    m = build_model("maxvit-nano", "pmal", IMG, NCLS, carry=True)
    m(torch.randn(2, 3, IMG, IMG), T={"stage3": 3}).sum().backward()
    st = m.s3
    assert st.gamma.grad is not None and st.beta.grad is not None and st.iter_emb.grad is not None
    assert st.block.block_att.alpha.grad is not None and st.block.grid_att.alpha.grad is not None   # M3
    assert st.block.grid_att.att.rel_table.grad[2].abs().sum() > 0                  # per-iteration table t=2
    assert st.block.mb.pw1.weight.grad.abs().sum() > 0


def test_truncated_bptt_with_tuple_state():
    m = build_model("maxvit-nano", "pmal", IMG, NCLS, carry=True, T={"stage3": 9})
    assert m.loop_spec["stage3"]["trunc_k"] == 4
    m(torch.randn(2, 3, IMG, IMG), T={"stage3": 12}).sum().backward()


def test_residual_trace():
    m = build_model("maxvit-nano", "pmal", IMG, NCLS).eval()
    _, tr = m(torch.randn(2, 3, IMG, IMG), T={"stage3": 4}, return_trace=True)
    assert len(tr["stage3"]) == 4 and len(tr["stage2"]) == 1


def test_overfit_random_T_and_recalibration():
    """Gradient-flow sanity (overfit_check.py): 16 random images, random-T training, then R13."""
    torch.manual_seed(0)
    random.seed(0)
    x, y = torch.randn(16, 3, IMG, IMG), torch.randint(0, NCLS, (16,))
    m = build_model("maxvit-nano", "pmal", IMG, NCLS, carry=True)
    opt = torch.optim.AdamW(m.parameters(), 2e-3)
    losses = []
    for _ in range(40):
        loss = nn.functional.cross_entropy(m(x, T=sample_T(m.random_T_spec())), y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    assert sum(losses[-5:]) / 5 < 0.1 * losses[0]
    recalibrate_bn(m, [x], T=m.train_T(), device="cpu")
    m.eval()
    acc = (m(x).argmax(1) == y).float().mean().item()
    assert acc >= 0.9, acc
