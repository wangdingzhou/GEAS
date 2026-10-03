"""FAD-v2: Frequency-Aware Decoder block v2.

Identity-safe Haar wavelet residual block, inserted at c2 decoder tap
(between SampleAdapter output and FRM input) in MM-DINO decoder.

Design:
  - Haar DWT/IDWT via fixed slicing (NO pytorch_wavelets dep)
  - Shared zero-init Conv1x1 applied to all 4 subbands (LL, LH, HL, HH)
  - Per-channel gamma gate (init=1.0), applied post-IDWT
  - Residual form: out = x + gamma * delta
  - Step 0 = bit-exact identity (gamma=1, delta=0 since Conv1x1 zero-init)

Param count at channels=256:
  Conv1x1 (no bias): channels^2 = 65,536
  gamma (per-channel): channels  =    256
  Total:                        = 65,792 (~66K)

Target: boundary bottleneck (-26pt D0.2 gap), NOT pathology.
D0.6 audit confirmed router-based frequency methods (Earth-Adapter)
cannot fix Building-vs-Rural pathology -- JS(B,R) approx 0 double evidence
(mean-weight JS 0.0003-0.0011, histogram JS 0.0003-0.0199, all 7 FFT blocks).
FAD-v2 avoids router entirely (channel-wise gamma only).

Why Haar over FFT:
  FSDENet (Fu 2025 JSTARS) + SFFNet (Yang 2024 TGRS) both verified Haar > FFT
  for RS boundary/texture on LoveDA/Potsdam/Vaihingen. FFT on coarse 32x32
  token grid (Earth-Adapter's failure mode) blurry; Haar clean 4-subband at
  c2 (B, 256, 128, 128) -> 4 x (B, 256, 64, 64).
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["FADv2Block", "haar_dwt", "haar_idwt"]


def haar_dwt(x: torch.Tensor) -> tuple:
    """Fixed Haar DWT, J=1, orthonormal (analysis matrix H_2D = (1/2) * 4x4 Haar).

    Args:
        x: (B, C, H, W) with H, W even.
    Returns:
        (LL, LH, HL, HH), each (B, C, H/2, W/2). Same dtype/device as input.

    Math:
        LL = 0.5 * (x00 + x01 + x10 + x11)   (low-low, average)
        LH = 0.5 * (x00 - x01 + x10 - x11)   (horizontal high, vertical low)
        HL = 0.5 * (x00 + x01 - x10 - x11)   (horizontal low, vertical high)
        HH = 0.5 * (x00 - x01 - x10 + x11)   (diagonal high)
      where x_ij = x[..., i::2, j::2].
    """
    H, W = x.shape[-2], x.shape[-1]
    if H % 2 != 0 or W % 2 != 0:
        raise ValueError(
            f"haar_dwt requires even H, W; got ({H}, {W})"
        )
    x00 = x[..., 0::2, 0::2]
    x01 = x[..., 0::2, 1::2]
    x10 = x[..., 1::2, 0::2]
    x11 = x[..., 1::2, 1::2]
    LL = 0.5 * (x00 + x01 + x10 + x11)
    LH = 0.5 * (x00 - x01 + x10 - x11)
    HL = 0.5 * (x00 + x01 - x10 - x11)
    HH = 0.5 * (x00 - x01 - x10 + x11)
    return LL, LH, HL, HH


def haar_idwt(
    LL: torch.Tensor,
    LH: torch.Tensor,
    HL: torch.Tensor,
    HH: torch.Tensor,
) -> torch.Tensor:
    """Inverse Haar DWT, J=1, orthonormal (H_2D symmetric so inverse = forward).

    Args:
        LL, LH, HL, HH: each (B, C, Hh, Wh).
    Returns:
        out: (B, C, 2*Hh, 2*Wh). Same dtype/device as inputs.

    Math (since H_2D^T = H_2D for Haar):
        x00 = 0.5 * (LL + LH + HL + HH)
        x01 = 0.5 * (LL - LH + HL - HH)
        x10 = 0.5 * (LL + LH - HL - HH)
        x11 = 0.5 * (LL - LH - HL + HH)
    """
    x00 = 0.5 * (LL + LH + HL + HH)
    x01 = 0.5 * (LL - LH + HL - HH)
    x10 = 0.5 * (LL + LH - HL - HH)
    x11 = 0.5 * (LL - LH - HL + HH)
    B, C, Hh, Wh = LL.shape
    out = torch.empty(
        B, C, Hh * 2, Wh * 2, device=LL.device, dtype=LL.dtype
    )
    out[..., 0::2, 0::2] = x00
    out[..., 0::2, 1::2] = x01
    out[..., 1::2, 0::2] = x10
    out[..., 1::2, 1::2] = x11
    return out


class FADv2Block(nn.Module):
    """Identity-safe Haar wavelet residual block for MM-DINO decoder c2.

    Args:
        channels: c2 channel count. Default 256 (matches MM-DINO
            adapter_out_channels[0]). For other taps (c3/c4/c5) use 512/1024/1024.
        zero_init: if True (default), Conv1x1 weights init to zero so step 0
            is bit-exact identity (since gamma init = 1.0 and delta = 0).

    Forward:
        x: (B, channels, H, W) -> (B, channels, H, W).
        H, W must be even. For MM-DINO c2 at image_size=512, H=W=128 (H/4).

    Param count at channels=256:
        Conv1x1 (no bias): channels^2 = 65,536
        gamma (per-channel): channels   =    256
        Total:                            = 65,792 (~66K)
    """

    def __init__(self, channels: int = 256, zero_init: bool = True):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.ones(channels))
        if zero_init:
            nn.init.zeros_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Haar DWT -> 4 subbands at half resolution
        LL, LH, HL, HH = haar_dwt(x)
        # Shared Conv1x1 applied to each subband (4 applications, 1 set of weights)
        delta_LL = self.conv(LL)
        delta_LH = self.conv(LH)
        delta_HL = self.conv(HL)
        delta_HH = self.conv(HH)
        # IDWT back to full resolution
        delta = haar_idwt(delta_LL, delta_LH, delta_HL, delta_HH)
        # Per-channel gamma gate, residual form
        return x + self.gamma.view(1, -1, 1, 1) * delta

    def extra_repr(self) -> str:
        return f"channels={self.channels}, params={sum(p.numel() for p in self.parameters()):,}"
