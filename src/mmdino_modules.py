"""MM-DINO architecture port (SampleAdapter + FRM + PRN + Decoder).

Ported from the MM-DINO official implementation (Decoder.py,
sample_adapter.py). RGB-only (num_modalities=1), so the SEFusion path is
dropped — this matches MM-DINO's RGB-only default.

References:
    - Decoder.py:354  — Decoder class (FRM + PRN + ConvBNReLU out_conv)
    - Decoder.py:29   — FeatureReinforcementModule (cross-scale reinforcement)
    - Decoder.py:245  — ProgressiveRefinementNeck (top-down + bottom-up)
    - sample_adapter.py:19 — SampleAdapter (token -> H/4..H/32 pyramid)

DINOv3 notes:
    MM-DINO assumes DINOv2-L (32x32 token grid). DINOv3 returns non-spatial
    tokens (with extra register/cls tokens), so we take outputs from
    `backbone.get_intermediate_layers(x, n=[4,11,17,23])` which already strips
    non-spatial tokens. We then reshape using (B, N, D) -> (B, D, patch_h, patch_w)
    via patch_h = H//16, patch_w = W//16.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Basic block
# ---------------------------------------------------------------------------

class ConvBNReLU(nn.Sequential):
    """Conv2d (no bias) + BatchNorm2d + ReLU(inplace)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        pad: int = 1,
    ) -> None:
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        super().__init__(*layers)


# ---------------------------------------------------------------------------
# Feature Reinforcement Module (FRM)
# ---------------------------------------------------------------------------

class FeatureFusionModule(nn.Module):
    """FRM's internal fusion block.

    Input: `c_fuse` (concatenated cross-scale features, channels = 4*mid_d)
           `c` (identity shortcut from the original scale, channels = in_d[scale])
    Output: out_d-channel feature.

    `drop_rate` is accepted to match MM-DINO's reference API but is unused —
    the reference code's `nn.Dropout(drop_rate)` line is commented out
    (Decoder.py:172-173).
    """

    def __init__(self, fuse_d: int, in_d: int, out_d: int, drop_rate: float = 0.0,
                 lambda_decayed: float = 1.0) -> None:
        super().__init__()
        self.drop_rate = drop_rate  # unused; matches MM-DINO reference signature
        self.fuse_d = fuse_d
        self.in_d = in_d
        self.out_d = out_d
        # Paper1284 CVPR 2026 free fold-in: decayed identity shortcut.
        # Formula: c_out = relu(c_fuse + lambda_decayed * conv_identity(c))
        # Default 1.0 = original MM-DINO behavior (no-op). P1 sets 0.65 per scale.
        self.lambda_decayed = float(lambda_decayed)
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(fuse_d, fuse_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(fuse_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(fuse_d, fuse_d, kernel_size=3, stride=1, padding=1, groups=fuse_d),
            nn.BatchNorm2d(fuse_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(fuse_d, out_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_d),
        )
        self.conv_identity = nn.Conv2d(in_d, out_d, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, c_fuse: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        c_fuse = self.conv_fuse(c_fuse)
        c_out = self.relu(c_fuse + self.lambda_decayed * self.conv_identity(c))
        return c_out


class FeatureReinforcementModule(nn.Module):
    """4-scale cross-reinforcement: for each scale s in {2,3,4,5}, pool all
    other scales down to s's resolution and aggregate.

    Inputs are 4 spatial feature maps (c2, c3, c4, c5) at H/4, H/8, H/16, H/32
    respectively. Outputs 4 features at the same resolutions.

    NOTE: this is a faithful port. The reference code indexes scales 2..5 with
    in_d[scale-1] lookup; we keep the same indexing.
    """

    def __init__(
        self,
        in_d: Sequence[int],     # length-5: [in_d_check, c2_d, c3_d, c4_d, c5_d]
        out_d: int = 256,
        drop_rate: float = 0.0,
        lambda_decayed: float = 1.0,
        ffm_class: type = FeatureFusionModule,
    ) -> None:
        """ffm_class: FeatureFusionModule subclass to use for aggregation.
        Defaults to FeatureFusionModule (MM-DINO behavior). WPD decoder passes
        WavFFM (src/wpd_decoder.py) for wavelet-based frequency-aware fusion."""
        super().__init__()
        # NOTE: in_d slot 0 is unused — mirrors MM-DINO's reference Decoder.py
        # where the caller passes [in_channels[0]] + in_channels (5 elements).
        # The aggregation lookups all use in_d[1..4]. Kept this way on purpose
        # for faithful porting; do not "clean up" without re-verifying against
        # the reference.
        if len(in_d) != 5:
            raise ValueError(
                f"FRM expects in_d of length 5 ([check, c2, c3, c4, c5]); got {len(in_d)}"
            )
        self.in_d = list(in_d)
        self.mid_d = out_d // 2
        self.out_d = out_d

        # For scales 2..5 (orig) x scales 2..5 (target): pool/identity -> 1x1 conv
        self.conv_scales = nn.ModuleDict()
        for orig_scale in range(2, 6):
            for target_scale in range(2, 6):
                key = f"conv_scale{target_scale}_c{orig_scale}"
                self.conv_scales[key] = self._create_conv_block(
                    self.in_d[orig_scale - 1],
                    self.mid_d,
                    scale=target_scale,
                    orig_scale=orig_scale,
                )

        # Per-scale fusion: 4 * mid_d -> out_d, with identity from the original scale.
        # lambda_decayed propagates to all 4 FFMs (Paper1284 free fold-in).
        # ffm_class allows swapping in WavFFM (wavelet) without touching FRM logic.
        self.conv_aggregation_s2 = ffm_class(self.mid_d * 4, self.in_d[1], out_d, drop_rate, lambda_decayed)
        self.conv_aggregation_s3 = ffm_class(self.mid_d * 4, self.in_d[2], out_d, drop_rate, lambda_decayed)
        self.conv_aggregation_s4 = ffm_class(self.mid_d * 4, self.in_d[3], out_d, drop_rate, lambda_decayed)
        self.conv_aggregation_s5 = ffm_class(self.mid_d * 4, self.in_d[4], out_d, drop_rate, lambda_decayed)

    def _create_conv_block(
        self,
        in_channels: int,
        mid_channels: int,
        scale: int,
        orig_scale: int,
    ) -> nn.Module:
        layers: list[nn.Module] = []
        if scale > orig_scale:
            # Pool down to target scale
            layers.append(
                nn.MaxPool2d(
                    kernel_size=2 ** (scale - orig_scale),
                    stride=2 ** (scale - orig_scale),
                )
            )
        if scale == orig_scale:
            layers.extend([
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            ])
        else:
            # 1x1 conv -> DW 3x3 (groups=mid_channels)
            layers.extend([
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    mid_channels,
                    mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=mid_channels,
                ),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            ])
        return nn.Sequential(*layers)

    def _process_scale(self, c: torch.Tensor, scale_idx: int) -> list[torch.Tensor]:
        """scale_idx: 0=c2, 1=c3, 2=c4, 3=c5. Returns 4 outputs at scales 2..5."""
        outputs = []
        for target in range(2, 6):
            key = f"conv_scale{target}_c{scale_idx + 2}"
            out = self.conv_scales[key](c)
            if target < scale_idx + 2:
                # Upsample back to the original scale (orig_scale > target)
                up = 2 ** (scale_idx + 2 - target)
                out = F.interpolate(out, scale_factor=(up, up), mode="bilinear", align_corners=False)
            outputs.append(out)
        return outputs

    def forward(
        self,
        c2: torch.Tensor,
        c3: torch.Tensor,
        c4: torch.Tensor,
        c5: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c2_scales = self._process_scale(c2, 0)
        c3_scales = self._process_scale(c3, 1)
        c4_scales = self._process_scale(c4, 2)
        c5_scales = self._process_scale(c5, 3)

        s2 = self.conv_aggregation_s2(
            torch.cat([c2_scales[0], c3_scales[0], c4_scales[0], c5_scales[0]], dim=1), c2
        )
        s3 = self.conv_aggregation_s3(
            torch.cat([c2_scales[1], c3_scales[1], c4_scales[1], c5_scales[1]], dim=1), c3
        )
        s4 = self.conv_aggregation_s4(
            torch.cat([c2_scales[2], c3_scales[2], c4_scales[2], c5_scales[2]], dim=1), c4
        )
        s5 = self.conv_aggregation_s5(
            torch.cat([c2_scales[3], c3_scales[3], c4_scales[3], c5_scales[3]], dim=1), c5
        )

        return s2, s3, s4, s5


# ---------------------------------------------------------------------------
# Progressive Refinement Neck (PRN)
# ---------------------------------------------------------------------------

class ProgressiveRefinementNeck(nn.Module):
    """Top-down + bottom-up refinement across 4 scales. Returns 3 scales
    (P2_out, P3_out, P4_out) — P5 is folded into P4 via downsampling.

    output_stage="p2_only" (Innovation 3 Tier-1.2, 2026-08-14): drops the
    degenerate output branch entirely and returns [P2_out] only. P4_out carried
    zero P4/P5 information (it is a conv of downsampled P2_refine); P2_out
    instead consumes the bottom-up P3 intermediate (P3_td1) directly.
    Removes out_convs_3/out_convs_4 (−1.77M params).

    resize_mode="unified": bilinear upsample + MaxPool
    downsample instead of nearest for every direction (original MM-DINO uses
    nearest for both).
    """

    def __init__(
        self,
        channels_list: Sequence[int],
        num_stages: int = 1,
        output_stage: str = "full",
        nonlinear: bool = False,
        resize_mode: str = "nearest",
    ) -> None:
        super().__init__()
        if output_stage not in ("full", "p2_only"):
            raise ValueError(
                f"Unknown output_stage={output_stage!r}; expected 'full' or 'p2_only'"
            )
        if resize_mode not in ("nearest", "unified"):
            raise ValueError(
                f"Unknown resize_mode={resize_mode!r}; expected 'nearest' or 'unified'"
            )
        if output_stage == "p2_only" and num_stages < 1:
            raise ValueError("p2_only requires num_stages >= 1 (P3_td1 must exist)")
        self.num_stages = num_stages
        self.output_stage = output_stage
        self.resize_mode = resize_mode
        cl = list(channels_list)

        def conv(c_in, c_out):
            """Optional BN+ReLU wrap on every bare PRN conv (ablation variant)."""
            m = nn.Conv2d(c_in, c_out, 3, padding=1)
            if nonlinear:
                return nn.Sequential(m, nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))
            return m

        # Top-down convs: (P5+P4)->P4, (P4+P3)->P3, (P3+P2)->P2
        self.td_convs = nn.ModuleList([
            conv(cl[i + 1] + cl[i], cl[i])
            for i in range(len(cl) - 1)
        ])

        # Bottom-up reuse convs (P2+P3)->P3, (P3+P2+P2)->P2
        self.reuse_convs = nn.ModuleList([
            conv(cl[0] + cl[1], cl[1]),
            conv(cl[1] + cl[0] * 2, cl[0]),
        ])

        # Output convs (project refined features back to feature channels)
        self.out_convs_2 = conv(cl[0] * 2 + cl[1], cl[0])
        if output_stage == "full":
            self.out_convs_3 = conv(cl[1] + cl[2], cl[1])
            self.out_convs_4 = conv(cl[0], cl[2])

    def _resize(self, x: torch.Tensor, size: torch.Size) -> torch.Tensor:
        if self.resize_mode == "unified":
            th, tw = int(size[0]), int(size[1])
            ih, iw = int(x.shape[-2]), int(x.shape[-1])
            if th <= ih and tw <= iw and ih % th == 0 and iw % tw == 0:
                return F.max_pool2d(x, kernel_size=(ih // th, iw // tw))
            return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return F.interpolate(x, size=size, mode="nearest")

    def forward(
        self,
        backbone_features: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        P2_in, P3_in, P4_in, P5_in = backbone_features

        # Top-down
        P5_td = P5_in
        P4_td = self.td_convs[2](torch.cat([self._resize(P5_td, P4_in.shape[-2:]), P4_in], dim=1))
        P3_td = self.td_convs[1](torch.cat([self._resize(P4_td, P3_in.shape[-2:]), P3_in], dim=1))
        P2_td = self.td_convs[0](torch.cat([self._resize(P3_td, P2_in.shape[-2:]), P2_in], dim=1))

        P2_refine = P2_td
        for _ in range(self.num_stages):
            P3_td1 = self.reuse_convs[0](
                torch.cat([self._resize(P2_refine, P3_in.shape[-2:]), P3_in], dim=1)
            )
            resize_ = self._resize(P3_td1, P2_in.shape[-2:])
            cat_feature = torch.cat([resize_, P2_td, P2_in], dim=1)
            P2_refine = self.reuse_convs[1](cat_feature)

        if self.output_stage == "p2_only":
            # Degenerate P4_out/P3_out output branch removed (P4_out is a
            # conv of downsampled P2_refine and carries no P4/P5 information;
            # the head only ever consumed P2_out). P2_out takes the bottom-up
            # P3 intermediate P3_td1 directly in place of nearest-up(P3_out).
            P2_out = self.out_convs_2(
                torch.cat([
                    self._resize(P3_td1, P2_in.shape[-2:]),
                    P2_refine,
                    P2_in,
                ], dim=1)
            )
            return [P2_out]

        P4_out = self.out_convs_4(self._resize(P2_refine, P4_in.shape[-2:]))
        P3_out = self.out_convs_3(
            torch.cat([self._resize(P4_out, P3_in.shape[-2:]), P3_in], dim=1)
        )
        P2_out = self.out_convs_2(
            torch.cat([self._resize(P3_out, P2_in.shape[-2:]), P2_refine, P2_in], dim=1)
        )

        return [P2_out, P3_out, P4_out]


# ---------------------------------------------------------------------------
# SampleAdapter — token -> spatial pyramid
# ---------------------------------------------------------------------------

class SampleAdapter(nn.Module):
    """Project ViT tokens at 4 taps to a spatial pyramid.

    MM-DINO uses taps at layers [4, 11, 17, 23] (0-indexed), all sharing the
    same `embed_dim` (1024 for ViT-L). Each tap is projected to a different
    channel count and resized to a different spatial resolution:
        tap 0 -> out_channels[0] (256) at H/4   (ConvTranspose k=4, s=4)
        tap 1 -> out_channels[1] (512) at H/8   (ConvTranspose k=2, s=2)
        tap 2 -> out_channels[2] (1024) at H/16 (Identity)
        tap 3 -> out_channels[3] (1024) at H/32 (Conv k=3, s=2, p=1)

    Note: tap 2 is at H/16 already (since ViT-L/16 outputs H/16 tokens), and
    Identity keeps it there. tap 3 is downsampled to H/32 by the stride-2
    Conv. So the 4 outputs are NOT at the resolutions listed above — they
    are at H/4, H/8, H/16, H/32. (The `out_channels` and resize choices are
    consistent: tap i -> H/(2^(6-i))).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        num_modalities: int = 1,
    ) -> None:
        super().__init__()
        if num_modalities != 1:
            raise NotImplementedError(
                "Baseline protocol is RGB-only. SEFusion path is dropped."
            )
        self.in_channels = in_channels
        self.out_channels = list(out_channels)
        self.num_modalities = num_modalities

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=oc,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            for oc in self.out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(self.out_channels[0], self.out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(self.out_channels[1], self.out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(self.out_channels[3], self.out_channels[3], kernel_size=3, stride=2, padding=1),
        ])

    def forward(
        self,
        features_list: Sequence[torch.Tensor],
        patch_h: int,
        patch_w: int,
    ) -> list[torch.Tensor]:
        """features_list: 4 ViT outputs of shape (B, N, D).

        Returns a list of 4 spatial feature maps at H/4, H/8, H/16, H/32.
        """
        if len(features_list) != 4:
            raise ValueError(
                f"SampleAdapter expects exactly 4 taps; got {len(features_list)}"
            )
        out: list[torch.Tensor] = []
        for i, x in enumerate(features_list):
            if x.dim() != 3:
                raise ValueError(
                    f"Tap {i} expected shape (B, N, D); got {tuple(x.shape)}"
                )
            B, N, D = x.shape
            if N != patch_h * patch_w:
                raise ValueError(
                    f"Tap {i} has N={N} tokens but patch_h*patch_w={patch_h*patch_w}. "
                    "Did the backbone strip non-spatial tokens?"
                )
            x = x.permute(0, 2, 1).reshape(B, D, patch_h, patch_w)
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        return out


# ---------------------------------------------------------------------------
# MM-DINO default Decoder = FRM + PRN + ConvBNReLU out_conv
# ---------------------------------------------------------------------------

class MMDINODecoder(nn.Module):
    """RGB-only default Decoder = FRM + PRN + out_conv.

    Single-modality path of MM-DINO `Decoder` class (Decoder.py:354). The
    SEFusion branch is dropped because our protocol is RGB-only.
    """

    def __init__(
        self,
        n_classes: int = 6,
        in_channels: Sequence[int] = (256, 512, 1024, 1024),
        out_channels: int = 256,
        decoder_out_channels: int | None = None,
        lambda_decayed: float = 1.0,
        prn_output_stage: str = "full",
        prn_resize_mode: str = "nearest",
        prn_nonlinear: bool = False,
    ) -> None:
        # Back-compat alias: older callers pass `decoder_out_channels`
        if decoder_out_channels is not None:
            out_channels = decoder_out_channels
        super().__init__()
        self.in_channels = list(in_channels)
        self.out_channels = out_channels

        # FRM expects in_d[1..4] = [c2, c3, c4, c5]; in_d[0] is unused (check slot)
        self.frm = FeatureReinforcementModule(
            in_d=[in_channels[0]] + list(in_channels),
            out_d=out_channels,
            lambda_decayed=lambda_decayed,
        )

        # PRN operates at the FRM output channel (out_d, all 4 scales)
        self.neck = ProgressiveRefinementNeck(
            channels_list=[out_channels] * 4,
            num_stages=1,
            output_stage=prn_output_stage,
            resize_mode=prn_resize_mode,
            nonlinear=prn_nonlinear,
        )

        # Final classifier head: out_d -> n_classes
        self.out_conv = ConvBNReLU(out_channels, n_classes, kernel_size=1, pad=0)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError(f"Decoder expects 4 features; got {len(features)}")
        c2, c3, c4, c5 = features
        s2, s3, s4, s5 = self.frm(c2, c3, c4, c5)
        p2 = self.neck([s2, s3, s4, s5])[0]
        return self.out_conv(p2)


# ---------------------------------------------------------------------------
# Full model — backbone + SampleAdapter + Decoder
# ---------------------------------------------------------------------------

class MMDINOModel(nn.Module):
    """Backbone (frozen) + SampleAdapter + MMDINODecoder.

    Forward path:
        x: (B, 3, H, W) -> backbone.get_intermediate_layers(x, n=[4,11,17,23])
        -> 4 token tensors of shape (B, N, D)
        -> SampleAdapter -> 4 spatial maps at H/4, H/8, H/16, H/32
        -> Decoder -> logits at H/4 resolution
        -> bilinear interpolate to (H, W) for loss
    """

    def __init__(
        self,
        backbone: nn.Module,
        n_classes: int = 6,
        adapter_out_channels: Sequence[int] = (256, 512, 1024, 1024),
        decoder_out_channels: int = 256,
        taps_0idx: Sequence[int] = (4, 11, 17, 23),
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.taps_0idx = list(taps_0idx)
        self.embed_dim = backbone.embed_dim
        self.adapter = SampleAdapter(
            in_channels=self.embed_dim,
            out_channels=adapter_out_channels,
            num_modalities=1,
        )
        self.decoder = MMDINODecoder(
            n_classes=n_classes,
            in_channels=adapter_out_channels,
            out_channels=decoder_out_channels,
        )
        self.patch_size = 16  # ViT-L/16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        # Returns list of (B, N, D) tensors, one per tap, with non-spatial
        # tokens (cls/register) already stripped.
        tokens_list = self.backbone.get_intermediate_layers(
            x, n=self.taps_0idx, reshape=False, norm=True,
        )

        spatial_features = self.adapter(tokens_list, patch_h=patch_h, patch_w=patch_w)
        logits = self.decoder(spatial_features)

        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits
