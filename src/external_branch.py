"""External Spatial Pathway — raw image / shallow tap branch + serial injection.

Two variants:
  - RI (primary): raw image conv stem (3→32→64→256) → 2 ResBlock (256→32→256
    bottleneck, Houlsby #3 internal skip) → zero-init output conv → LayerScale γ.
    Inject serial: p2_enhanced = p2 + γ ⊙ branch(raw_image).
  - ST (ablation): DINOv3 layer-1 tap (H/16 grid) → Linear(1024, 256) →
    reshape + ConvTranspose(256, 256, k=4, s=4) to H/4 → same ResBlock + output.

Design (Houlsby 7-condition checklist):
  #1 near-identity init: output conv weight=0 + bias=0; γ init=1e-4 (not 0, avoid grad death)
  #2 serial injection (not parallel concat) — Houlsby negative result
  #3 internal skip in ResBlock (pre-activation residual)
  #4 bottleneck ratio ~8:1 (256→32→256), param budget ~0.53M
  #5 zero-init γ gate (NOT sigmoid gate, per feedback-identity-init-poison-for-gating)
  #6 single-factor: gate open (A) vs γ≡0 (B) is the only variable
  #7 main run = RI; ST is Variant C ablation

Build via build_external_branch(config) factory. Returns None when disabled.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ============================================================================
# Building blocks
# ============================================================================

class LayerScale(nn.Module):
    """Per-channel learnable scale γ, init near-zero (Houlsby #5).

    Shape: (1, C, 1, 1) for 4D feature maps. Init to `init_value` (default 1e-4).
    NOT a sigmoid gate — sigmoid with bias>=0 is poison for gating modules
    (per feedback-identity-init-poison-for-gating: bias>=0 → sigmoid≈0.5-1 →
    gradient death or gate-locked). LayerScale is a path module (safe).

    Variant A: γ trainable, init=1e-4.
    Variant B: γ ≡ 0 (frozen), identity sanity check.
    Variant D: γ ≡ 1.0 (fixed, ablation only).
    """

    def __init__(self, dim: int, init_value: float = 1e-4, freeze_value: Optional[float] = None):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), float(init_value)))
        if freeze_value is not None:
            with torch.no_grad():
                self.gamma.fill_(float(freeze_value))
            self.gamma.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class ResBlockBottleneck(nn.Module):
    """Pre-activation residual block with 8:1 bottleneck (256 → 32 → 256).

    Internal skip (Houlsby #3): out = x + f(x) where f = conv1×1 → BN → ReLU
    → conv3×3 → BN → ReLU → conv1×1.

    Args:
        channels: external channels (256 default, matches p2 embed_dim)
        bottleneck_ratio: 8 (i.e. 256→32→256); 4 = 256→64→256
    """

    def __init__(self, channels: int = 256, bottleneck_ratio: int = 8):
        super().__init__()
        mid = max(channels // bottleneck_ratio, 16)
        self.conv1 = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        # ReLU is module-level (no learned params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        return identity + out


def _zero_init_last_conv(block: nn.Module) -> None:
    """Houlsby #1: zero-init the final conv weight + BN so block ≈ identity at start.

    For ResBlockBottleneck: zero conv3 weight + zero bn3 weight + zero bn3 bias.
    """
    if hasattr(block, "conv3") and hasattr(block, "bn3"):
        nn.init.zeros_(block.conv3.weight)
        if block.conv3.bias is not None:
            nn.init.zeros_(block.conv3.bias)
        nn.init.zeros_(block.bn3.weight)
        nn.init.zeros_(block.bn3.bias)


# ============================================================================
# Variant RI — raw image conv stem (primary)
# ============================================================================

class ExternalBranchRI(nn.Module):
    """Variant RI: raw image → conv stem → H/4 → 2 ResBlock → zero-init output.

    Pipeline:
        x: (B, 3, H, W)
        Stem: Conv(3→32, k3, s2) → Conv(32→64, k3, s1)             → (B, 64, H/2, W/2)
        Down: Conv(64→256, k3, s2)                                  → (B, 256, H/4, W/4)
        Refine: ResBlock(256→32→256) × 2                            → (B, 256, H/4, W/4)
        Output: Conv(256→256, k1) zero-init weight + zero-init bias → (B, 256, H/4, W/4)

    Trainable params: 167k stem + 295k ResBlock + 65.5k output ≈ 0.53M.

    Variant D (γ=1.0 ablation): same branch, but γ in parent decoder is forced to 1.0.
    Variant E (parallel concat ablation): handled at decoder injection site, not here.
    """

    def __init__(self, out_channels: int = 256):
        super().__init__()
        # Stem (stride-2 once → H/2)
        self.stem_conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.stem_bn1 = nn.BatchNorm2d(32)
        self.stem_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.stem_bn2 = nn.BatchNorm2d(64)
        # Down to H/4
        self.down_conv = nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.down_bn = nn.BatchNorm2d(out_channels)
        # 2 ResBlocks with 8:1 bottleneck
        self.res1 = ResBlockBottleneck(out_channels, bottleneck_ratio=8)
        self.res2 = ResBlockBottleneck(out_channels, bottleneck_ratio=8)
        # Output 1×1 conv — Houlsby #1 zero-init
        self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)

        # Apply near-identity init to all "final" layers
        _zero_init_last_conv(self.res1)
        _zero_init_last_conv(self.res2)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, raw_image: torch.Tensor) -> torch.Tensor:
        """raw_image: (B, 3, H, W) → branch output (B, out_channels, H/4, W/4)."""
        x = F.relu(self.stem_bn1(self.stem_conv1(raw_image)), inplace=True)
        x = F.relu(self.stem_bn2(self.stem_conv2(x)), inplace=True)
        x = F.relu(self.down_bn(self.down_conv(x)), inplace=True)
        x = self.res1(x)
        x = self.res2(x)
        x = self.out_conv(x)  # zero-init ⇒ x ≈ 0 at start (Houlsby #1)
        return x


# ============================================================================
# Variant ST — DINOv3 shallow tap (ablation)
# ============================================================================

class ExternalBranchST(nn.Module):
    """Variant ST: DINOv3 layer-1 tap (H/16 grid) → project → upsample to H/4.

    Used for Variant C ablation (raw-image vs shallow-tap information source comparison).
    Not the main run.

    Pipeline:
        tap: (B, N=1024, D=1024) from DINOv3.get_intermediate_layers(x, n=[1]) reshape=True
             → (B, 1024, 32, 32) for 512² input
        Project: Linear(1024, 256) → reshape (B, 256, 32, 32)
        Upsample: ConvTranspose2d(256, 256, k=4, s=4) → (B, 256, 128, 128) = H/4
        Refine + output: same as RI (2 ResBlock + zero-init out_conv)
    """

    def __init__(self, in_dim: int = 1024, out_channels: int = 256):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_channels)
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels,
                                           kernel_size=4, stride=4, padding=0)
        self.upsample_bn = nn.BatchNorm2d(out_channels)
        self.res1 = ResBlockBottleneck(out_channels, bottleneck_ratio=8)
        self.res2 = ResBlockBottleneck(out_channels, bottleneck_ratio=8)
        self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)

        _zero_init_last_conv(self.res1)
        _zero_init_last_conv(self.res2)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, shallow_tap: torch.Tensor) -> torch.Tensor:
        """shallow_tap: (B, N, D) → (B, out_channels, H/4, W/4).

        Caller must pass DINOv3 layer-1 tap with N = (H/16)², D = embed_dim.
        """
        B, N, D = shallow_tap.shape
        # Infer spatial dims from N (assumes square)
        s = int(N ** 0.5)
        assert s * s == N, f"Tap N={N} not a perfect square (expected (H/16)²)"
        x = self.proj(shallow_tap)                          # (B, N, out_channels)
        x = x.permute(0, 2, 1).reshape(B, -1, s, s)         # (B, C, H/16, W/16)
        x = F.relu(self.upsample_bn(self.upsample(x)), inplace=True)  # (B, C, H/4, W/4)
        x = self.res1(x)
        x = self.res2(x)
        x = self.out_conv(x)
        return x


# ============================================================================
# Factory
# ============================================================================

def build_external_branch(config: dict) -> Optional[nn.Module]:
    """Factory: build external branch from config dict.

    Config schema (in cfg['decoder']['external_branch']):
        enabled: bool (default False)
        variant: 'RI' | 'ST' (default 'RI')
        out_channels: int (default 256, matches decoder embed_dim)
        gamma_init: float (default 1e-4; Variant D uses 1.0)
        gamma_freeze_value: float | None (default None; Variant B uses 0.0)
        st_in_dim: int (default 1024, DINOv3 ViT-L embed_dim)

    Returns None when disabled (Variant B no-branch control also returns None — the
    γ≡0 case is handled by the parent decoder, not by building a branch + freezing).
    """
    if not config or not config.get("enabled", False):
        return None

    variant = config.get("variant", "RI").upper()
    out_channels = int(config.get("out_channels", 256))

    if variant == "RI":
        branch = ExternalBranchRI(out_channels=out_channels)
    elif variant == "ST":
        st_in_dim = int(config.get("st_in_dim", 1024))
        branch = ExternalBranchST(in_dim=st_in_dim, out_channels=out_channels)
    else:
        raise ValueError(
            f"Unknown external_branch.variant: {variant!r}; expected 'RI' or 'ST'"
        )

    n_params = sum(p.numel() for p in branch.parameters())
    print(
        f"[ExternalBranch] variant={variant}, out_channels={out_channels}, "
        f"trainable_params={n_params:,} (~{n_params/1e6:.2f}M)"
    )
    return branch


def build_layer_scale(config: dict, dim: int) -> Optional[LayerScale]:
    """Factory: build γ LayerScale from config.

    Returns None when branch disabled (no injection).
    Variant A: trainable γ init=1e-4.
    Variant B: γ≡0 frozen (identity sanity).
    Variant D: γ≡1.0 frozen (no-scale ablation).
    """
    if not config or not config.get("enabled", False):
        return None

    init_value = float(config.get("gamma_init", 1e-4))
    freeze_value = config.get("gamma_freeze_value", None)

    ls = LayerScale(dim=dim, init_value=init_value,
                    freeze_value=freeze_value)
    is_frozen = freeze_value is not None
    print(
        f"[LayerScale γ] dim={dim}, init={init_value:.4g}, "
        f"freeze_value={freeze_value} ({'frozen' if is_frozen else 'trainable'})"
    )
    return ls


# ============================================================================
# Self-test (smoke)
# ============================================================================

def _selftest() -> None:
    """Quick shape + identity-init sanity check (call via python -c)."""
    torch.manual_seed(0)

    # RI: shape + near-zero output at init
    branch_ri = ExternalBranchRI(out_channels=256)
    x = torch.randn(2, 3, 512, 512)
    out = branch_ri(x)
    assert out.shape == (2, 256, 128, 128), f"RI shape: {tuple(out.shape)}"
    out_norm = out.norm().item()
    print(f"✓ RI forward: shape {tuple(out.shape)}, L2 norm {out_norm:.4e} "
          f"(should be ~0 from zero-init out_conv)")

    n_params_ri = sum(p.numel() for p in branch_ri.parameters())
    print(f"✓ RI param count: {n_params_ri:,} (~{n_params_ri/1e6:.2f}M, spec target ~0.53M)")

    # ST: shape
    branch_st = ExternalBranchST(in_dim=1024, out_channels=256)
    tap = torch.randn(2, 1024, 1024)  # 32×32 grid for 512² input
    out_st = branch_st(tap)
    assert out_st.shape == (2, 256, 128, 128), f"ST shape: {tuple(out_st.shape)}"
    print(f"✓ ST forward: shape {tuple(out_st.shape)}, L2 norm {out_st.norm().item():.4e}")

    # LayerScale γ
    ls = LayerScale(dim=256, init_value=1e-4)
    feat = torch.randn(2, 256, 128, 128)
    out_ls = ls(feat)
    assert torch.allclose(out_ls, feat * 1e-4)
    print(f"✓ LayerScale γ=1e-4 scales correctly (max input {feat.abs().max():.3f} → "
          f"max output {out_ls.abs().max():.4e})")

    # γ ≡ 0 frozen (Variant B sanity)
    ls_zero = LayerScale(dim=256, init_value=0.0, freeze_value=0.0)
    out_zero = ls_zero(feat)
    assert out_zero.abs().max().item() < 1e-10
    assert not ls_zero.gamma.requires_grad
    print(f"✓ Variant B γ≡0 frozen: output max abs {out_zero.abs().max().item():.2e}, "
          f"requires_grad={ls_zero.gamma.requires_grad}")

    # γ ≡ 1.0 frozen (Variant D ablation)
    ls_one = LayerScale(dim=256, init_value=1.0, freeze_value=1.0)
    out_one = ls_one(feat)
    assert torch.allclose(out_one, feat)
    assert not ls_one.gamma.requires_grad
    print(f"✓ Variant D γ≡1.0 frozen: identity pass-through, requires_grad={ls_one.gamma.requires_grad}")

    print("\n=== ALL EXTERNAL BRANCH SELF-TESTS PASSED ===")


if __name__ == "__main__":
    _selftest()
