"""SF-Adapter: spatial + frequency + adaptive gate.

Design summary:
- Spatial branch: 3 parallel dilated DWConv3x3 (dilations 1/2/3) + 1x1 PWConv fusion.
- Frequency branch: FFT8Branch using FULL fft2/ifft2 (NOT rfft2) and symmetric
  (H,W) radial band masks; per-channel per-band complex gains zero-init.
- Adaptive sigmoid gate (MLP on pooled spa/freq/cross) mixing spa and freq.
- ReZero / LayerScale gamma (init=1e-3) and small-kaiming up_proj (gain=0.01).
- adapter_disable flag short-circuits to identity for the parity check.

Integration is non-invasive: HALoRASeg registers forward hooks on
backbone.blocks[18:24] (the last 6 blocks, i.e. blocks 19-24 in 1-indexed).
The hook splits the (B, 1+n_storage+N_patches, C) residual stream into
non-spatial tokens (CLS + storage) and patch tokens, applies the adapter to the
patch tokens only, then re-attaches the non-spatial tokens.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Radial band masks (Issue 6: full fft2 + symmetric (H,W) mask)
# ---------------------------------------------------------------------------

def _build_radial_band_masks(H: int, W: int, n_bands: int) -> torch.Tensor:
    """Build n_bands symmetric radial ring masks covering all (H, W) freq pixels.

    Bands are concentric rings centered at (H//2, W//2). Each pixel belongs to
    exactly one band, so sum over bands = all-ones (complete coverage).
    Returns tensor of shape (n_bands, H, W) with values in {0.0, 1.0}.
    """
    cy, cx = H // 2, W // 2
    ys = torch.arange(H, dtype=torch.float32) - cy
    xs = torch.arange(W, dtype=torch.float32) - cx
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    r = torch.sqrt(grid_y ** 2 + grid_x ** 2)
    max_r = math.sqrt(float(cy * cy + cx * cx))
    if max_r <= 0:
        band_idx = torch.zeros_like(r, dtype=torch.long)
    else:
        band_idx = (r / max_r * n_bands).long().clamp(0, n_bands - 1)
    masks = torch.zeros(n_bands, H, W, dtype=torch.float32)
    for b in range(n_bands):
        masks[b] = (band_idx == b).float()
    return masks


# ---------------------------------------------------------------------------
# FFT8 branch
# ---------------------------------------------------------------------------

class FFT8Branch(nn.Module):
    """Full fft2/ifft2 + radial 8-band complex gains per channel.

    Output: x_freq with same shape as x, zero at init (since gain_real and
    gain_imag are both zero-init). At init, the branch contributes nothing,
    so the SF adapter's residual output is purely from the spatial branch.

    Forward: x (B,H,W,C) -> freq-filtered x (B,H,W,C).
    """

    def __init__(self, channels: int, n_bands: int = 8):
        super().__init__()
        self.channels = channels
        self.n_bands = n_bands
        # Per-channel per-band complex gain, zero-init for identity-at-start.
        self.gain_real = nn.Parameter(torch.zeros(channels, n_bands))
        self.gain_imag = nn.Parameter(torch.zeros(channels, n_bands))
        # Lazy mask cache (not a parameter; recomputed if shape changes).
        self._cached_masks: Optional[torch.Tensor] = None
        self._cached_hw: Optional[Tuple[int, int]] = None

    def _get_masks(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        if (
            self._cached_masks is None
            or self._cached_hw != (H, W)
            or self._cached_masks.device != device
        ):
            masks = _build_radial_band_masks(H, W, self.n_bands).to(device)
            self._cached_masks = masks
            self._cached_hw = (H, W)
        return self._cached_masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C). FFT2 is numerically unstable in fp16/autocast, so
        # run the whole forward in fp32 with autocast disabled.
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            B, H, W, C = x.shape
            assert C == self.channels, f"channels mismatch: got {C}, expected {self.channels}"
            masks = self._get_masks(H, W, x.device)  # (n_bands, H, W)

            # Move to (B, C, H, W) for fft2 over the last two dims.
            x_nchw = x.permute(0, 3, 1, 2).contiguous()
            X = torch.fft.fft2(x_nchw)  # full complex (B, C, H, W)
            X = torch.fft.fftshift(X, dim=(-2, -1))  # DC to center

            # Per-channel per-pixel complex gain: einsum on real/imag separately
            # (older PyTorch einsum does not support complex operands).
            #   gain_real (C, n_bands) x masks (n_bands, H, W) -> (C, H, W)
            gain_real_pix = torch.einsum("cn,nHW->cHW", self.gain_real, masks)
            gain_imag_pix = torch.einsum("cn,nHW->cHW", self.gain_imag, masks)
            pixel_gain = torch.complex(gain_real_pix, gain_imag_pix)  # (C, H, W) complex

            X = X * pixel_gain.unsqueeze(0)  # broadcast over B
            X = torch.fft.ifftshift(X, dim=(-2, -1))
            x_freq = torch.fft.ifft2(X).real  # (B, C, H, W)

            return x_freq.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)


# ---------------------------------------------------------------------------
# Late SF-Adapter
# ---------------------------------------------------------------------------

class LateSFAdapter(nn.Module):
    """Late SF-Adapter: spatial + frequency + adaptive sigmoid gate.

    Forward (when adapter_disable=False):
        h   = down_proj(x)                         # (B,H,W,d)
        spa = pwconv(cat[dwconv1(h), dwconv2(h), dwconv3(h)])
        frq = fft8(h)                              # (B,H,W,d), zero at init
        g   = sigmoid(mlp([pool(spa), pool(frq), pool(spa*frq)]))  # (B,1)
        a   = g*spa + (1-g)*frq
        out = x + gamma * up_proj(a)               # (B,H,W,C)

    With adapter_disable=True, forward returns x unchanged (parity check).
    """

    def __init__(
        self,
        in_channels: int = 1024,
        bottleneck: int = 256,
        n_bands: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.bottleneck = bottleneck
        self.n_bands = n_bands

        # Down / up projections.
        self.down_proj = nn.Linear(in_channels, bottleneck)
        self.up_proj = nn.Linear(bottleneck, in_channels)
        # Small-kaiming init for up_proj (gain=0.01).
        nn.init.kaiming_normal_(
            self.up_proj.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu"
        )
        nn.init.zeros_(self.up_proj.bias)

        # Spatial: 3 parallel dilated DWConv3x3 + 1x1 PWConv fusion.
        # Operate on (B, d, H, W); dilations 1/2/3 -> pads 1/2/3.
        self.dwconv1 = nn.Conv2d(
            bottleneck, bottleneck, 3, padding=1, dilation=1, groups=bottleneck, bias=True
        )
        self.dwconv2 = nn.Conv2d(
            bottleneck, bottleneck, 3, padding=2, dilation=2, groups=bottleneck, bias=True
        )
        self.dwconv3 = nn.Conv2d(
            bottleneck, bottleneck, 3, padding=3, dilation=3, groups=bottleneck, bias=True
        )
        self.pwconv = nn.Conv2d(bottleneck * 3, bottleneck, 1, bias=True)

        # Frequency branch.
        self.fft8 = FFT8Branch(bottleneck, n_bands=n_bands)

        # Adaptive gate MLP (3d -> d -> 1).
        self.gate_mlp = nn.Sequential(
            nn.Linear(bottleneck * 3, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, 1),
        )
        # Zero-init last linear so initial gate = sigmoid(0) = 0.5 (balanced).
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)

        # ReZero / LayerScale gamma (init=1e-3 per spec).
        self.gamma = nn.Parameter(torch.tensor(1e-3))

        # Flag toggled externally for the parity check.
        self.adapter_disable = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C) -> (B, H, W, C).

        When adapter_disable=True, returns x unchanged (for parity test).

        The whole forward runs in fp32 with autocast disabled. The adapter is
        small (~5.6M params total) so the perf hit is negligible, and this
        avoids FFT2 / sigmoid fp16 instabilities that produce NaN.
        """
        if self.adapter_disable:
            return x

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            B, H, W, C = x.shape
            assert C == self.in_channels, f"C mismatch: got {C}, expected {self.in_channels}"

            # Down-project: (B,H,W,C) -> (B,H,W,d).
            h = self.down_proj(x)

            # Spatial branch: (B,H,W,d) -> (B,d,H,W) -> concat -> PW -> (B,H,W,d).
            h_nchw = h.permute(0, 3, 1, 2).contiguous()
            s1 = self.dwconv1(h_nchw)
            s2 = self.dwconv2(h_nchw)
            s3 = self.dwconv3(h_nchw)
            spa_nchw = self.pwconv(torch.cat([s1, s2, s3], dim=1))  # (B,d,H,W)
            spa = spa_nchw.permute(0, 2, 3, 1).contiguous()  # (B,H,W,d)

            # Frequency branch: (B,H,W,d) -> (B,H,W,d), zero at init.
            frq = self.fft8(h)

            # Adaptive gate from pooled features.
            spa_pool = spa.mean(dim=(1, 2))  # (B,d)
            frq_pool = frq.mean(dim=(1, 2))  # (B,d)
            cross_pool = spa_pool * frq_pool  # (B,d)
            gate_in = torch.cat([spa_pool, frq_pool, cross_pool], dim=-1)  # (B,3d)
            gate = torch.sigmoid(self.gate_mlp(gate_in))  # (B,1)

            # Mix spa and freq via gate, then up-project.
            g = gate.view(B, 1, 1, 1)
            a = g * spa + (1.0 - g) * frq  # (B,H,W,d)
            out = self.up_proj(a)  # (B,H,W,C)

            return x + self.gamma * out


# ---------------------------------------------------------------------------
# Hook installation
# ---------------------------------------------------------------------------

def make_sf_block_hook(adapter: "LateSFAdapter", n_non_spatial: int):
    """Build a forward hook that applies `adapter` to a DINOv3 block's output.

    Block output is (B, 1+n_storage+N_patches, C). The hook:
      1. Splits off the first n_non_spatial tokens (CLS + storage) unchanged.
      2. Reshapes the patch tokens to (B, H, W, C) assuming H == W.
      3. Applies the adapter (residual, gamma-scaled).
      4. Restores the (B, T, C) layout.
    """

    def hook(module: nn.Module, inputs, output):
        # output may be a tensor or a tuple (tensor, ...). Handle both.
        if isinstance(output, tuple):
            x = output[0]
            rest = output[1:]
        else:
            x = output
            rest = None

        if not torch.is_tensor(x):
            return output  # unexpected; leave unchanged

        if x.dim() != 3:
            return output  # not (B,T,C); leave unchanged

        B, T, C = x.shape
        N = T - n_non_spatial
        if N <= 0:
            return output  # nothing spatial; leave unchanged

        hw = int(math.isqrt(N))
        if hw * hw != N:
            return output  # non-square; leave unchanged

        x_ns = x[:, :n_non_spatial, :]
        x_sp = x[:, n_non_spatial:, :]
        x_4d = x_sp.reshape(B, hw, hw, C)

        x_4d_out = adapter(x_4d)
        x_sp_out = x_4d_out.reshape(B, N, C)
        x_out = torch.cat([x_ns, x_sp_out], dim=1)

        if rest is not None:
            return (x_out,) + rest
        return x_out

    return hook


def install_sf_adapters(
    model: nn.Module,
    block_indices_0idx: List[int],
    in_channels: int = 1024,
    bottleneck: int = 256,
    n_bands: int = 8,
) -> List["torch.utils.hooks.RemovableHandle"]:
    """Install LateSFAdapter hooks on the given backbone block indices.

    The adapters are stored on `model` as `model.sf_adapters` (a ModuleList) so
    they participate in .to(device), state_dict, and optimizer construction.
    Returns the list of hook handles (for later removal if needed).

    `n_non_spatial` is read from model.n_storage_tokens.
    """
    if not hasattr(model, "backbone"):
        raise AttributeError("model has no .backbone attribute")
    if not hasattr(model, "n_storage_tokens"):
        raise AttributeError("model has no .n_storage_tokens attribute")

    n_blocks_total = len(model.backbone.blocks)
    for idx in block_indices_0idx:
        if not (0 <= idx < n_blocks_total):
            raise IndexError(
                f"block idx {idx} out of range (have {n_blocks_total} blocks)"
            )

    n_non_spatial = 1 + int(model.n_storage_tokens)
    adapters = nn.ModuleList(
        [
            LateSFAdapter(
                in_channels=in_channels,
                bottleneck=bottleneck,
                n_bands=n_bands,
            )
            for _ in block_indices_0idx
        ]
    )
    if hasattr(model, "sf_adapters"):
        raise RuntimeError("model already has sf_adapters; uninstall first")
    model.sf_adapters = adapters
    model._sf_block_indices = list(block_indices_0idx)

    handles = []
    for adapter, idx in zip(adapters, block_indices_0idx):
        block = model.backbone.blocks[idx]
        handle = block.register_forward_hook(
            make_sf_block_hook(adapter, n_non_spatial)
        )
        handles.append(handle)
    model._sf_hook_handles = handles
    return handles


def uninstall_sf_adapters(model: nn.Module) -> None:
    """Remove hooks installed by install_sf_adapters; delete the ModuleList."""
    handles = getattr(model, "_sf_hook_handles", None)
    if handles:
        for h in handles:
            h.remove()
        model._sf_hook_handles = []
    if hasattr(model, "sf_adapters"):
        del model.sf_adapters
    if hasattr(model, "_sf_block_indices"):
        del model._sf_block_indices


def set_sf_adapter_disable(model: nn.Module, value: bool) -> None:
    """Toggle adapter_disable on all installed SF adapters."""
    if not hasattr(model, "sf_adapters"):
        return
    for adapter in model.sf_adapters:
        adapter.adapter_disable = bool(value)
