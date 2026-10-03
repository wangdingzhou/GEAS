"""
GEAS: parameter-efficient adaptation of a frozen DINOv3 backbone
for remote-sensing semantic segmentation
Model implementation for remote sensing semantic segmentation.
"""

import math
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

DINOV3_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dinov3")
if DINOV3_DIR not in sys.path:
    sys.path.insert(0, DINOV3_DIR)

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

_REPO_ROOT = os.path.dirname(_SRC_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# DINOv3 backbone builder
# ---------------------------------------------------------------------------

def _load_checkpoint_file(checkpoint: str) -> Dict[str, torch.Tensor]:
    """Load a PyTorch or safetensors checkpoint into a CPU state dict."""
    if checkpoint.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Loading .safetensors checkpoints requires the safetensors package."
            ) from exc
        return load_file(checkpoint, device="cpu")

    try:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    return state_dict


def _is_hf_dinov3_state_dict(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any(k.startswith("embeddings.") or k.startswith("layer.") for k in state_dict)


def _allowed_hf_missing_key(key: str) -> bool:
    return (
        key == "rope_embed.periods"
        or key.startswith("local_cls_norm.")
        or key.endswith(".attn.qkv.bias_mask")
    )


def _convert_hf_dinov3_state_dict(
    state_dict: Dict[str, torch.Tensor],
    model_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Convert HuggingFace/ModelScope DINOv3 ViT keys to the local Meta keys."""
    converted: Dict[str, torch.Tensor] = {}
    consumed = set()

    def put(src: str, dst: str) -> None:
        if src in state_dict:
            converted[dst] = state_dict[src]
            consumed.add(src)

    put("embeddings.cls_token", "cls_token")
    put("embeddings.mask_token", "mask_token")
    put("embeddings.register_tokens", "storage_tokens")
    put("embeddings.patch_embeddings.weight", "patch_embed.proj.weight")
    put("embeddings.patch_embeddings.bias", "patch_embed.proj.bias")
    put("norm.weight", "norm.weight")
    put("norm.bias", "norm.bias")

    block_ids = sorted(
        {
            int(match.group(1))
            for key in state_dict
            for match in [re.match(r"layer\.(\d+)\.", key)]
            if match is not None
        }
    )
    for idx in block_ids:
        src_prefix = f"layer.{idx}"
        dst_prefix = f"blocks.{idx}"
        for suffix in ("weight", "bias"):
            q_key = f"{src_prefix}.attention.q_proj.{suffix}"
            k_key = f"{src_prefix}.attention.k_proj.{suffix}"
            v_key = f"{src_prefix}.attention.v_proj.{suffix}"
            if suffix == "bias" and q_key in state_dict and v_key in state_dict and k_key not in state_dict:
                converted[f"{dst_prefix}.attn.qkv.{suffix}"] = torch.cat(
                    [state_dict[q_key], torch.zeros_like(state_dict[q_key]), state_dict[v_key]], dim=0
                )
                consumed.update({q_key, v_key})
                continue

            has_any = any(k in state_dict for k in (q_key, k_key, v_key))
            has_all = all(k in state_dict for k in (q_key, k_key, v_key))
            if has_any and not has_all:
                raise KeyError(f"Incomplete HF q/k/v projection for block {idx} {suffix}")
            if has_all:
                converted[f"{dst_prefix}.attn.qkv.{suffix}"] = torch.cat(
                    [state_dict[q_key], state_dict[k_key], state_dict[v_key]], dim=0
                )
                consumed.update({q_key, k_key, v_key})

        put(f"{src_prefix}.attention.o_proj.weight", f"{dst_prefix}.attn.proj.weight")
        put(f"{src_prefix}.attention.o_proj.bias", f"{dst_prefix}.attn.proj.bias")
        put(f"{src_prefix}.norm1.weight", f"{dst_prefix}.norm1.weight")
        put(f"{src_prefix}.norm1.bias", f"{dst_prefix}.norm1.bias")
        put(f"{src_prefix}.norm2.weight", f"{dst_prefix}.norm2.weight")
        put(f"{src_prefix}.norm2.bias", f"{dst_prefix}.norm2.bias")
        put(f"{src_prefix}.mlp.up_proj.weight", f"{dst_prefix}.mlp.fc1.weight")
        put(f"{src_prefix}.mlp.up_proj.bias", f"{dst_prefix}.mlp.fc1.bias")
        put(f"{src_prefix}.mlp.down_proj.weight", f"{dst_prefix}.mlp.fc2.weight")
        put(f"{src_prefix}.mlp.down_proj.bias", f"{dst_prefix}.mlp.fc2.bias")
        put(f"{src_prefix}.layer_scale1.lambda1", f"{dst_prefix}.ls1.gamma")
        put(f"{src_prefix}.layer_scale2.lambda1", f"{dst_prefix}.ls2.gamma")

    unexpected = sorted(set(state_dict) - consumed)
    if unexpected:
        raise KeyError(f"Unexpected HF DINOv3 keys: {unexpected[:20]}")

    missing = sorted(k for k in model_state if k not in converted and not _allowed_hf_missing_key(k))
    if missing:
        raise KeyError(f"HF DINOv3 conversion did not produce required keys: {missing[:20]}")

    full_state = dict(model_state)
    for key in full_state:
        if key.endswith(".attn.qkv.bias_mask"):
            full_state[key] = torch.zeros_like(full_state[key])

    for key, value in converted.items():
        if key not in model_state:
            raise KeyError(f"Converted key not present in model: {key}")
        if tuple(value.shape) != tuple(model_state[key].shape):
            if value.numel() == model_state[key].numel():
                value = value.reshape_as(model_state[key])
            else:
                raise ValueError(
                    f"Shape mismatch for {key}: checkpoint {tuple(value.shape)} "
                    f"!= model {tuple(model_state[key].shape)}"
                )
        full_state[key] = value
    return full_state


def _build_dinov3_backbone(name: str, checkpoint: Optional[str] = None) -> nn.Module:
    """Build DINOv3 ViT backbone and optionally load pretrained weights."""
    from dinov3.hub.backbones import dinov3_vits16, dinov3_vitb16, dinov3_vitl16, Weights

    # ViT-L SAT493M needs weights=Weights.SAT493M so the factory sets
    # untie_global_and_local_cls_norm=True internally.
    extra_kwargs = {}
    if name == "vitl16" and checkpoint and "sat493m" in checkpoint.lower():
        extra_kwargs["weights"] = Weights.SAT493M

    factories = {"vits16": dinov3_vits16, "vitb16": dinov3_vitb16, "vitl16": dinov3_vitl16}
    if name not in factories:
        raise ValueError(f"Unknown backbone: {name}. Choose from {list(factories.keys())}")

    model = factories[name](pretrained=False, **extra_kwargs)

    if checkpoint:
        state_dict = _load_checkpoint_file(checkpoint)
        if _is_hf_dinov3_state_dict(state_dict):
            state_dict = _convert_hf_dinov3_state_dict(state_dict, model.state_dict())
            print("Converted HuggingFace/ModelScope DINOv3 checkpoint to local key format")
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded DINOv3 {name} weights from {checkpoint}")
    else:
        print(
            f"WARNING: DINOv3 {name} initialized with random weights. "
            f"Set backbone.checkpoint in config to load pretrained weights."
        )

    return model


# ---------------------------------------------------------------------------
# Multi-Scale DW-Conv for Structural Path Enhancement
# ---------------------------------------------------------------------------

class MultiScaleDWConv(nn.Module):
    """Multi-scale depthwise separable convolution (dilation={1,2,3}).

    Replaces single DW-Conv in SSD-LoRA structural path.
    Different dilation rates capture different spatial frequencies:
        d=1: local edges/textures (high-freq, RF 3x3)
        d=2: medium structures (mid-freq, RF 7x7)
        d=3: large context (low-freq, RF 11x11)
    """

    def __init__(self, channels: int, dilations: tuple = (1, 2, 3)):
        super().__init__()
        self.n_scales = len(dilations)
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=d, dilation=d, groups=channels, bias=False)
            for d in dilations
        ])
        self.pw = nn.Conv2d(channels * self.n_scales, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [dw(x) for dw in self.dw_convs]
        return self.pw(torch.cat(outs, dim=1))




# ---------------------------------------------------------------------------
# Texture Channel Attention Module (TCAM) for Structural Path
# ---------------------------------------------------------------------------

class TCAM(nn.Module):
    """Texture Channel Attention Module.

    Computes per-sample, per-channel scaling from token covariance/Gram.
    Used to modulate SSD-LoRA structural path's spatial tokens.

    Input:  h2d of shape (B, r, H, W)
    Output: per-channel scale s of shape (B, r, 1, 1)

    Paths:
      - use_mlp=True:
          centered_cov or normalized_gram -> diag -> log1p -> MLP -> s
          s = 1 + gamma * tanh(mlp_output)
      - use_mlp=False (control):
          s = 1 (identity; isolates MLP+gate contribution; dwconv still applied)

    diag computed in fp32 to preserve magnitude in mixed-precision training.
    No per-sample standardization: keeps absolute texture energy for
    hard/easy patch interpretability.
    """

    def __init__(
        self,
        channels: int,
        tcam_type: str = "cov",        # "cov" (centered) or "gram" (non-centered)
        use_mlp: bool = True,
        gamma: float = 0.2,
        hidden_min: int = 8,
    ):
        super().__init__()
        self.channels = channels
        self.tcam_type = tcam_type
        self.use_mlp = use_mlp
        self.gamma = gamma
        self.hidden_min = hidden_min

        if use_mlp:
            hidden = max(hidden_min, 2 * channels)
            self.mlp = nn.Sequential(
                nn.Linear(channels, hidden),
                nn.GELU(),
                nn.Linear(hidden, channels),
            )
            # zero-init last layer: initial s = 1 + gamma*tanh(0) = 1
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h2d: torch.Tensor) -> torch.Tensor:
        """h2d: (B, r, H, W) -> s: (B, r, 1, 1)"""
        B, r, H, W = h2d.shape

        if not self.use_mlp:
            return torch.ones(B, r, 1, 1, device=h2d.device, dtype=h2d.dtype)

        # Disable autocast inside TCAM: stats and MLP both run in fp32 for stability.
        # If module was .half()ed (e.g. pure-fp16 inference), cast MLP weights to fp32.
        weight_dtype = next(self.mlp.parameters()).dtype
        with torch.autocast(device_type=h2d.device.type, enabled=False):
            h2d_fp32 = h2d.float()
            h_flat = h2d_fp32.reshape(B, r, H * W)

            if self.tcam_type == "cov":
                mean = h_flat.mean(dim=2, keepdim=True)          # (B, r, 1)
                centered = h_flat - mean                          # (B, r, N)
                diag = (centered ** 2).mean(dim=2)                # (B, r) per-channel variance
            elif self.tcam_type == "gram":
                diag = (h_flat ** 2).mean(dim=2)                  # (B, r) per-channel energy
            else:
                raise ValueError(f"Unknown tcam_type: {self.tcam_type}")

            # log1p preserves magnitude, compresses dynamic range
            diag = torch.log1p(diag.clamp(min=0))

            mlp = self.mlp.float() if weight_dtype != torch.float32 else self.mlp
            s = mlp(diag)                                        # (B, r)
            s = 1.0 + self.gamma * torch.tanh(s)                 # (B, r)

        return s.view(B, r, 1, 1).to(h2d.dtype)


# ---------------------------------------------------------------------------
# SSD-LoRA Module
# ---------------------------------------------------------------------------

class TCAMGAug(nn.Module):
    """TCAM with Gaussian spatial supplement (GAug).

    Inspired by LocAtViT GAug (Tang ICLR 2026).
    GAug adds a learned per-token Gaussian-shape bias to attention logits before
    softmax; we adapt the idea to TCAM (channel attention) by producing a
    per-batch 2D Gaussian mask that multiplicatively supplements the channel
    scaling.

    Output: (B, r, H, W) per-pixel modulation = s_channel * s_spatial
      where:
        s_channel  (B, r, 1, 1) — same as original TCAM, range ~[1-gamma, 1+gamma]
        s_spatial  (B, 1, H, W) — 1 + gamma_spa * alpha * exp(-0.5 * d²/var)

    Identity-init: gaug_proj last layer is zero-init → alpha=0 → s_spatial=1
    at start, so output = s_channel (matches original TCAM at init).

    Notes:
      - var is shared per-batch (one isotropic-ish 2D Gaussian per sample).
        LocAtViT predicts per-token var; we predict per-sample to keep params
        low (only 3 extra outputs per SSD-LoRA module).
      - The Gaussian is always centered at the grid center; LocAtViT's
        gaussian_2d_numerator computes a full N×N pairwise matrix which is
        not directly applicable to channel attention.
    """

    def __init__(
        self,
        channels: int,
        tcam_type: str = "cov",
        gamma: float = 0.2,
        hidden_min: int = 8,
        gamma_spa: float = 0.1,
    ):
        super().__init__()
        self.channels = channels
        self.gamma = gamma
        self.gamma_spa = gamma_spa

        # Channel scaling (original TCAM)
        self.channel_tcam = TCAM(
            channels,
            tcam_type=tcam_type,
            use_mlp=True,
            gamma=gamma,
            hidden_min=hidden_min,
        )

        # GAug spatial supplement: predict per-batch (log_var_y, log_var_x, raw_alpha)
        # from GAP of h2d. Single linear layer (3 outputs) keeps params minimal.
        self.gaug_proj = nn.Linear(channels, 3, bias=True)
        nn.init.zeros_(self.gaug_proj.weight)
        nn.init.zeros_(self.gaug_proj.bias)

    def forward(self, h2d: torch.Tensor) -> torch.Tensor:
        """h2d: (B, r, H, W) -> scale (B, r, H, W)."""
        B, r, H, W = h2d.shape
        s_channel = self.channel_tcam(h2d)  # (B, r, 1, 1)

        with torch.autocast(device_type=h2d.device.type, enabled=False):
            h2d_fp32 = h2d.float()
            h_gap = h2d_fp32.mean(dim=(2, 3))  # (B, r)
            gaug_out = self.gaug_proj(h_gap)  # (B, 3)
            log_var_y = gaug_out[:, 0:1]
            log_var_x = gaug_out[:, 1:2]
            raw_alpha = gaug_out[:, 2:3]
            var_y = F.softplus(log_var_y) + 1e-4  # (B, 1)
            var_x = F.softplus(log_var_x) + 1e-4  # (B, 1)
            # Identity-init: alpha(0)=0 via softplus(x) - softplus(0).
            # Plain softplus(0)=log(2)≈0.693 ≠ 0, which would give a non-trivial
            # Gaussian boost at init. The shift makes alpha=0 at init →
            # s_spatial = 1 + gamma_spa * 0 * gaussian = 1 (true identity).
            softplus_zero = math.log(2.0)
            alpha = F.softplus(raw_alpha) - softplus_zero  # (B, 1), =0 at init

            cy = (H - 1) / 2.0
            cx = (W - 1) / 2.0
            ys = torch.arange(H, device=h2d.device, dtype=torch.float32).view(H, 1)
            xs = torch.arange(W, device=h2d.device, dtype=torch.float32).view(1, W)
            d_y = (ys - cy) ** 2  # (H, 1)
            d_x = (xs - cx) ** 2  # (1, W)

            var_y = var_y.view(B, 1, 1)
            var_x = var_x.view(B, 1, 1)
            gaussian = torch.exp(
                -0.5 * (d_y.view(1, H, 1) / var_y + d_x.view(1, 1, W) / var_x)
            )  # (B, H, W)
            gaussian = gaussian.unsqueeze(1)  # (B, 1, H, W)

            alpha_view = alpha.view(B, 1, 1, 1)
            s_spatial = 1.0 + self.gamma_spa * alpha_view * gaussian  # (B, 1, H, W)

        # Cast back to h2d.dtype (matches original TCAM at line 302). Without
        # this, fp16 * fp32 promotes to fp32 and the downstream torch.cat
        # ([fp16_non_spatial, fp32_spatial]) silently promotes too, leaving
        # SSD-LoRA output in fp32 and forfeiting AMP speedup.
        return (s_channel * s_spatial).to(h2d.dtype)




class SSDLoRAModule(nn.Module):
    """
    Semantic-Structural Decoupled LoRA module.

    output = W_frozen(x)
           + alpha_sem * A_sem @ (B_sem @ x)
           + alpha_str * A_str @ spatial_path(B_str @ x)

    The structural path applies a spatial processor to the spatial (patch)
    tokens only; CLS and storage tokens bypass it. Available processors:

      - structural_path_type="single":  DWConv3x3 + Conv1x1 (legacy baseline)
      - structural_path_type="multi_scale_dwconv": dilation {1,2,3}
      - structural_path_type="conv_lora_moe": Conv-LoRA MoE (Zhong ICLR 2024),
        n experts with upsample_ratios [1..n], top-K=1 hard routing. Exposes
        ``self.last_moe_loss`` (load-balancing loss, scaled by loss_coef) on
        every forward; consumed by HALoRASeg.get_aux_losses().

    Aux-loss contract:
      self.last_moe_loss is a scalar buffer. Set to 0 at init, refreshed on
      every forward when conv_lora_moe is active, stays 0 otherwise.
    """

    def __init__(
        self,
        frozen_linear: nn.Linear,
        r_sem: int,
        r_str: int,
        n_non_spatial: int = 5,
        lora_mode: str = "ssd",
        structural_path_type: str = "single",
        use_rslora: bool = False,
        tcam_type: Optional[str] = None,
        tcam_gamma: float = 0.2,
        tcam_hidden_min: int = 8,
        # Conv-LoRA MoE knobs (only used when structural_path_type="conv_lora_moe")
        conv_lora_n_experts: int = 8,
        conv_lora_top_k: int = 1,
        conv_lora_upsample_ratios: Optional[list] = None,
        conv_lora_loss_coef: float = 1e-2,
    ):
        super().__init__()
        self.frozen_linear = frozen_linear
        self.r_sem = r_sem
        self.r_str = r_str
        self.n_non_spatial = n_non_spatial
        self.lora_mode = lora_mode
        self.structural_path_type = structural_path_type
        self.use_rslora = use_rslora
        self.tcam_type = tcam_type
        self.tcam_gamma = tcam_gamma
        self.tcam_hidden_min = tcam_hidden_min

        # Aux-loss slot is NOT mirrored here. Conv-LoRA MoE owns last_moe_loss
        # directly (see src/conv_lora.py). HALoRASeg.get_aux_losses() iterates
        # ConvLoRAMoE instances specifically to avoid double-counting that
        # would happen if both wrapper (SSDLoRAModule) and wrapped
        # (ConvLoRAMoE) carried the same attribute.
        # rsLoRA scaling factors precomputed (alpha / sqrt(r))
        self._rslora_scale_sem = 1.0 / math.sqrt(r_sem) if (use_rslora and r_sem > 0) else 1.0
        self._rslora_scale_str = 1.0 / math.sqrt(r_str) if (use_rslora and r_str > 0) else 1.0

        d_in = frozen_linear.in_features
        d_out = frozen_linear.out_features

        # Expose attributes that downstream code may access
        self.in_features = frozen_linear.in_features
        self.out_features = frozen_linear.out_features

        # Freeze original weights
        self.frozen_linear.weight.requires_grad_(False)
        if self.frozen_linear.bias is not None:
            self.frozen_linear.bias.requires_grad_(False)

        # Semantic path: standard LoRA
        if r_sem > 0:
            self.lora_sem_A = nn.Parameter(torch.empty(d_out, r_sem))
            self.lora_sem_B = nn.Parameter(torch.zeros(r_sem, d_in))
            nn.init.kaiming_uniform_(self.lora_sem_A, a=math.sqrt(5))
            self.alpha_sem = nn.Parameter(torch.ones(1))

        # Structural path: LoRA with spatial processing
        if r_str > 0:
            self.lora_str_A = nn.Parameter(torch.empty(d_out, r_str))
            self.lora_str_B = nn.Parameter(torch.zeros(r_str, d_in))
            nn.init.kaiming_uniform_(self.lora_str_A, a=math.sqrt(5))
            self.alpha_str = nn.Parameter(torch.ones(1))

            if lora_mode in ("ssd", "conv"):
                if structural_path_type == "multi_scale_dwconv":
                    self.spatial_conv = MultiScaleDWConv(r_str, dilations=(1, 2, 3))
                elif structural_path_type == "conv_lora_moe":
                    # Conv-LoRA MoE (Zhong ICLR 2024, AutoGluon port). See src/conv_lora.py.
                    from conv_lora import ConvLoRAMoE
                    self.spatial_conv = ConvLoRAMoE(
                        channels=r_str,
                        n_experts=conv_lora_n_experts,
                        top_k=conv_lora_top_k,
                        upsample_ratios=conv_lora_upsample_ratios,
                        loss_coef=conv_lora_loss_coef,
                    )
                else:
                    # "single" base path; also serves as TCAM preprocessing
                    self.spatial_conv = nn.Sequential(
                        nn.Conv2d(r_str, r_str, 3, padding=1, groups=r_str, bias=False),
                        nn.Conv2d(r_str, r_str, 1, bias=False),
                    )

                # TCAM (optional; only when this block is selected)
                if tcam_type is not None:
                    if tcam_type.endswith("_ln"):
                        raise NotImplementedError(
                            f"TCAM variant '{tcam_type}' is reserved for later, "
                            f"not implemented in this round"
                        )
                    if tcam_type == "tcam_gaug":
                        # Phase 1 GAug-TCAM (LocAtViT-inspired). Uses cov variant
                        # internally for channel scaling; adds Gaussian spatial
                        # supplement via TCAMGAug.
                        self.tcam = TCAMGAug(
                            r_str,
                            tcam_type="cov",
                            gamma=tcam_gamma,
                            hidden_min=tcam_hidden_min,
                        )
                    else:
                        if tcam_type.endswith("_nomlp"):
                            use_mlp_flag = False
                            base = tcam_type[:-len("_nomlp")]
                        else:
                            use_mlp_flag = True
                            base = tcam_type
                        if base == "tcam_cov":
                            cov_or_gram = "cov"
                        elif base == "tcam_gram":
                            cov_or_gram = "gram"
                        else:
                            raise ValueError(f"Unknown tcam_type base: {base}")
                        self.tcam = TCAM(
                            r_str,
                            tcam_type=cov_or_gram,
                            use_mlp=use_mlp_flag,
                            gamma=tcam_gamma,
                            hidden_min=tcam_hidden_min,
                        )
                else:
                    self.tcam = None

    def _apply_spatial_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial convolution to spatial (patch) tokens only."""
        B, N, r = x.shape
        n_ns = self.n_non_spatial
        non_spatial = x[:, :n_ns]
        spatial = x[:, n_ns:]

        h = w = int(math.sqrt(spatial.shape[1]))
        spatial = spatial.reshape(B, h, w, r).permute(0, 3, 1, 2)
        spatial = self.spatial_conv(spatial)
        spatial = spatial.permute(0, 2, 3, 1).reshape(B, h * w, r)

        return torch.cat([non_spatial, spatial], dim=1)

    def _apply_spatial_conv_with_tcam(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial conv + TCAM per-channel modulation to spatial tokens."""
        B, N, r = x.shape
        n_ns = self.n_non_spatial
        non_spatial = x[:, :n_ns]
        spatial = x[:, n_ns:]

        N_spatial = spatial.shape[1]
        side = int(math.sqrt(N_spatial))
        assert side * side == N_spatial, (
            f"Spatial tokens {N_spatial} not a perfect square; "
            f"check image_size and patch_size"
        )

        spatial_2d = spatial.reshape(B, side, side, r).permute(0, 3, 1, 2)

        # 1. spatial processor (single / multi_scale_dwconv / conv_lora_moe)
        # ConvLoRA MoE stores its load-balancing loss on self.spatial_conv
        # (ConvLoRAMoE instance); HALoRASeg.get_aux_losses() collects it.
        spatial_2d = self.spatial_conv(spatial_2d)

        # 2. TCAM per-channel scaling (s = 1 + gamma * tanh(mlp(diag)))
        if self.tcam is not None:
            scale = self.tcam(spatial_2d)                  # (B, r, 1, 1) or (B, r, H, W) for GAug
            spatial_2d = spatial_2d * scale

        spatial_out = spatial_2d.permute(0, 2, 3, 1).reshape(B, N_spatial, r)
        return torch.cat([non_spatial, spatial_out], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.frozen_linear(x)

        if self.r_sem > 0:
            sem = (x @ self.lora_sem_B.T) @ self.lora_sem_A.T
            result = result + (self.alpha_sem * self._rslora_scale_sem) * sem

        if self.r_str > 0:
            str_out = x @ self.lora_str_B.T
            if self.lora_mode in ("ssd", "conv"):
                if self.tcam is not None:
                    str_out = self._apply_spatial_conv_with_tcam(str_out)
                else:
                    str_out = self._apply_spatial_conv(str_out)
            str_out = str_out @ self.lora_str_A.T
            result = result + (self.alpha_str * self._rslora_scale_str) * str_out

        return result


# ---------------------------------------------------------------------------
# High-source spectral gate (FeatureSpectralGate)
# ---------------------------------------------------------------------------
#
# Design constraints:
#   - Source MUST be x BEFORE LoRA_B compression (full D-dimensional feature),
#     NOT low-rank r_spe=2/4 FFT of compressed LoRA response maps.
#   - FFT is computed on the spatial-token 2D grid (B, D, H, W) in fp32.
#   - Output is a per-sample, per-channel gate (B, D) used to modulate E_spe(x).
#   - Last MLP layer is zero-init so the gate starts at 0 → does not disturb
#     R0 init (gamma_spe also starts at 0).

class FeatureSpectralGate(nn.Module):
    """High-source spectral gate: operates on x BEFORE LoRA_B compression.

    Computes 2D FFT magnitudes on the spatial-token grid, pools into n_bands
    radial frequency bands per channel, then maps to a per-channel gate via
    an MLP whose last layer is zero-init by default.

    Input:  x of shape (B, N, D) where N = n_non_spatial + H*W
    Output: gate of shape (B, D) — multiply x by gate.unsqueeze(1) before
            passing to E_spe LoRA path.

    Dead-branch avoidance (Run A-main override §"Initialization"):
      When gamma_spe_init == 0 in the parent RoutedSSDLoRAModule, gate output
      is forced to 0 (zero_init_gate=True) so spectral forward contribution
      is exactly 0 at init. WARNING: this configuration is a dead branch —
      gamma_spe=0 ⇒ dL/d(spe)=0 ⇒ no spectral parameter receives gradient
      until gamma_spe is manually unfrozen.

      When gamma_spe_init > 0 (e.g., 1e-3, the default), zero_init_gate=False
      so the last MLP layer is small Kaiming (scaled 0.01). Gate output is
      small but non-zero, so all spectral params learn from step 0. Forward
      contribution is bounded by gamma_spe_init (~1e-3 scale).
    """

    def __init__(
        self,
        d_in: int,
        n_bands: int = 4,
        hidden_dim: int = 64,
        zero_init_gate: bool = True,
    ):
        super().__init__()
        self.d_in = d_in
        self.n_bands = n_bands

        # MLP from per-band magnitude stats to per-channel gate.
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_in * n_bands, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_in),
        )
        if zero_init_gate:
            # Silent mode: gate = 0 ⇒ E_spe contribution = 0 at start.
            # Combined with gamma_spe=0, this is a dead branch (override's
            # conservative option).
            nn.init.zeros_(self.gate_mlp[-1].weight)
            nn.init.zeros_(self.gate_mlp[-1].bias)
        else:
            # Active mode: small Kaiming so gate is non-zero at start.
            # Forward contribution is bounded by parent's gamma_spe_init.
            nn.init.kaiming_uniform_(self.gate_mlp[-1].weight, a=math.sqrt(5))
            nn.init.zeros_(self.gate_mlp[-1].bias)
            with torch.no_grad():
                self.gate_mlp[-1].weight.mul_(0.01)

    def _radial_band_masks(self, H: int, W: int, device: torch.device) -> List[torch.Tensor]:
        """Return n_bands boolean masks of shape (H, W) splitting the FFT
        magnitude grid into concentric radial frequency bands (low → high).
        """
        cy = (H - 1) / 2.0
        cx = (W - 1) / 2.0
        ys = torch.arange(H, device=device, dtype=torch.float32).view(H, 1).expand(H, W)
        xs = torch.arange(W, device=device, dtype=torch.float32).view(1, W).expand(H, W)
        r = torch.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
        r_max = r.max().clamp(min=1.0)
        r_norm = r / r_max
        # Band edges: [0, 1/n, 2/n, ..., 1]
        edges = torch.linspace(0.0, 1.0, self.n_bands + 1, device=device)
        masks = [(r_norm >= edges[i]) & (r_norm < edges[i + 1]) for i in range(self.n_bands)]
        # Last band is inclusive on the right edge so we don't lose the corner pixel.
        masks[-1] = masks[-1] | (r_norm == edges[-1])
        return masks

    def forward(self, x: torch.Tensor, n_non_spatial: int) -> torch.Tensor:
        """x: (B, N, D) -> gate: (B, D)."""
        B, N, D = x.shape
        spatial = x[:, n_non_spatial:]  # (B, H*W, D)
        N_spatial = spatial.shape[1]
        side = int(math.sqrt(N_spatial))
        if side * side != N_spatial:
            raise ValueError(
                f"FeatureSpectralGate requires square spatial grid, "
                f"got N_spatial={N_spatial}"
            )

        spatial_2d = spatial.reshape(B, side, side, D).permute(0, 3, 1, 2)  # (B, D, H, W)

        # FFT in fp32 for numerical stability under AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_fp32 = spatial_2d.float()
            # 2D FFT, shift origin to center for radial banding.
            freq = torch.fft.fftshift(torch.fft.fft2(x_fp32), dim=(-2, -1))
            mag = freq.abs()  # (B, D, H, W)

            masks = self._radial_band_masks(side, side, x.device)
            band_stats = []
            for mask in masks:
                # Per-channel mean magnitude within this band.
                mask_b = mask.view(1, 1, side, side).expand(B, D, -1, -1)
                band_mag = (mag * mask_b.float()).sum(dim=(-2, -1)) / mask_b.float().sum(dim=(-2, -1)).clamp(min=1.0)
                # log1p compresses dynamic range.
                band_stats.append(torch.log1p(band_mag.clamp(min=0)))  # (B, D)

        band_concat = torch.cat(band_stats, dim=-1)  # (B, D * n_bands)
        gate = self.gate_mlp(band_concat)  # (B, D)
        return gate


# ---------------------------------------------------------------------------
# Expert router (2*sigmoid, NOT softmax)
# ---------------------------------------------------------------------------
#
# Per Run A-main override §"Router":
#   - Independent residual scales s_* = 2 * sigmoid(router_logits_*)
#   - NOT softmax — experts do NOT compete for fixed probability mass.
#   - Last layer zero-init → logits=0 → s_* = 2*sigmoid(0) = 1.0 at start.
#   - At init, output ≡ R0 SSD-LoRA with TCAM (sem + spa + tex combined).
#   - Output shape: (B, n_experts) where n_experts is 3 (sem/spa/tex) or 4
#     (with high-source spectral).

class ExpertRouter(nn.Module):
    """Per-sample independent expert scale router (NOT softmax)."""

    def __init__(
        self,
        d_in: int,
        n_experts: int = 3,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.d_in = d_in
        self.n_experts = n_experts

        self.pool_proj = nn.Linear(d_in, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, n_experts)
        # Zero-init out_proj → logits = 0 → scales = 2*sigmoid(0) = 1.0.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, n_non_spatial: int) -> torch.Tensor:
        """x: (B, N, D) -> scales: (B, n_experts), all init to 1.0."""
        B, N, D = x.shape
        # Pool spatial tokens only; CLS/storage tokens carry no spatial structure.
        spatial = x[:, n_non_spatial:]  # (B, N_spatial, D)
        pooled = spatial.mean(dim=1)  # (B, D)
        h = torch.relu(self.pool_proj(pooled))  # (B, hidden_dim)
        logits = self.out_proj(h)  # (B, n_experts)
        scales = 2.0 * torch.sigmoid(logits)  # (B, n_experts), init=1.0
        return scales


# ---------------------------------------------------------------------------
# Routed SF-SSD-LoRA Module (Run A-main, 4-expert per override)
# ---------------------------------------------------------------------------
#
# Per Run A-main override §"Architecture":
#
#   y = frozen_linear(x)
#     + s_sem     * alpha_sem * E_sem(x)
#     + s_spa     * alpha_str * E_spa(x)
#     + s_tex     * alpha_str * (E_tex(x) - E_spa(x))
#     + gamma_spe * s_spe     * E_spe_high_source(x)
#
# Where:
#   E_sem:  standard semantic LoRA, x -> lora_sem_B -> lora_sem_A
#   E_spa:  structural WITHOUT TCAM, x -> lora_str_B -> spatial_conv -> lora_str_A
#   E_tex:  structural WITH TCAM,    x -> lora_str_B -> spatial_conv -> TCAM -> lora_str_A
#   (E_tex - E_spa): texture marginal — isolates TCAM contribution so router
#                    can scale it without disturbing the R0 structural base.
#   E_spe_high_source: optional; source from x BEFORE LoRA_B compression via
#                      FeatureSpectralGate. NOT r_spe=2/4 low-rank FFT.
#
# Init (router zero-init + gamma_spe=0): output ≡ R0 SSD-LoRA with TCAM.

class RoutedSSDLoRAModule(nn.Module):
    """Routed SF-SSD-LoRA: 4-expert per-sample gated SSD-LoRA (Run A-main)."""

    def __init__(
        self,
        frozen_linear: nn.Linear,
        r_sem: int,
        r_str: int,
        n_non_spatial: int = 5,
        structural_path_type: str = "single",
        use_rslora: bool = False,
        tcam_type: Optional[str] = None,
        tcam_gamma: float = 0.2,
        tcam_hidden_min: int = 8,
        spectral_enabled: bool = False,
        r_spe: int = 0,
        spectral_n_bands: int = 4,
        spectral_hidden_dim: int = 64,
        gamma_spe_init: float = 1e-3,
        router_hidden_dim: int = 64,
    ):
        super().__init__()
        self.frozen_linear = frozen_linear
        self.r_sem = r_sem
        self.r_str = r_str
        self.n_non_spatial = n_non_spatial
        self.lora_mode = "routed_sf_ssd"
        self.use_rslora = use_rslora
        self.tcam_type = tcam_type
        self.tcam_gamma = tcam_gamma
        self.tcam_hidden_min = tcam_hidden_min

        self._rslora_scale_sem = 1.0 / math.sqrt(r_sem) if (use_rslora and r_sem > 0) else 1.0
        self._rslora_scale_str = 1.0 / math.sqrt(r_str) if (use_rslora and r_str > 0) else 1.0
        self._rslora_scale_spe = 1.0 / math.sqrt(r_spe) if (use_rslora and r_spe > 0) else 1.0

        d_in = frozen_linear.in_features
        d_out = frozen_linear.out_features
        self.in_features = d_in
        self.out_features = d_out

        # Freeze original weights
        self.frozen_linear.weight.requires_grad_(False)
        if self.frozen_linear.bias is not None:
            self.frozen_linear.bias.requires_grad_(False)

        # --- Semantic expert (E_sem) ---
        if r_sem > 0:
            self.lora_sem_A = nn.Parameter(torch.empty(d_out, r_sem))
            self.lora_sem_B = nn.Parameter(torch.zeros(r_sem, d_in))
            nn.init.kaiming_uniform_(self.lora_sem_A, a=math.sqrt(5))
            self.alpha_sem = nn.Parameter(torch.ones(1))

        # --- Spatial + Texture experts (shared str params; TCAM only for tex) ---
        if r_str > 0:
            self.lora_str_A = nn.Parameter(torch.empty(d_out, r_str))
            self.lora_str_B = nn.Parameter(torch.zeros(r_str, d_in))
            nn.init.kaiming_uniform_(self.lora_str_A, a=math.sqrt(5))
            self.alpha_str = nn.Parameter(torch.ones(1))

            # Shared spatial_conv between E_spa and E_tex paths.
            if structural_path_type == "multi_scale_dwconv":
                self.spatial_conv = MultiScaleDWConv(r_str, dilations=(1, 2, 3))
            else:
                self.spatial_conv = nn.Sequential(
                    nn.Conv2d(r_str, r_str, 3, padding=1, groups=r_str, bias=False),
                    nn.Conv2d(r_str, r_str, 1, bias=False),
                )

            # TCAM is OPTIONAL in routed mode.
            # - With TCAM (shallow blocks): tex expert = E_tex - E_spa marginal
            # - Without TCAM (deep blocks): no tex expert; sem/spa/(spe) only
            if tcam_type is None:
                self.tcam = None
            else:
                if tcam_type.endswith("_ln"):
                    raise NotImplementedError(
                        f"TCAM variant '{tcam_type}' is reserved for later, "
                        f"not implemented in this round"
                    )
                if tcam_type.endswith("_nomlp"):
                    use_mlp_flag = False
                    base = tcam_type[:-len("_nomlp")]
                else:
                    use_mlp_flag = True
                    base = tcam_type
                if base == "tcam_cov":
                    cov_or_gram = "cov"
                elif base == "tcam_gram":
                    cov_or_gram = "gram"
                else:
                    raise ValueError(f"Unknown tcam_type base: {base}")
                self.tcam = TCAM(
                    r_str,
                    tcam_type=cov_or_gram,
                    use_mlp=use_mlp_flag,
                    gamma=tcam_gamma,
                    hidden_min=tcam_hidden_min,
                )

        # --- High-source spectral expert (E_spe, optional) ---
        self.spectral_enabled = spectral_enabled
        if spectral_enabled:
            if r_spe <= 0:
                raise ValueError("spectral_enabled requires r_spe > 0")
            # Separate A/B from sem/str — high-source rank on full D.

            # Dead-branch avoidance (override §"Initialization" allows gamma_spe
            # = 0 OR 1e-3): if all three of (gate_MLP_last_layer, lora_spe_B,
            # gamma_spe) are zero at init, then spe = 0 AND d(loss)/d(spe)=0 AND
            # all upstream gradients are 0 → spectral params never learn.

            # When spectral is enabled, we init gamma_spe > 0 (default 1e-3) AND
            # break the zero-init on the gate MLP last layer AND on lora_spe_B.
            # This produces a small but non-zero spe at init (~1e-3 scale on
            # spectral contribution), so sem/spa/tex stay ≈ R0 while spectral
            # learns from step 0. R0 drift is bounded by gamma_spe_init.
            self.lora_spe_A = nn.Parameter(torch.empty(d_out, r_spe))
            self.lora_spe_B = nn.Parameter(torch.empty(r_spe, d_in))
            nn.init.kaiming_uniform_(self.lora_spe_A, a=math.sqrt(5))
            if gamma_spe_init > 0:
                # Non-zero B init: small Kaiming so spe has a non-zero path at init.
                nn.init.kaiming_uniform_(self.lora_spe_B, a=math.sqrt(5))
                # Scale down so contribution stays small (gamma_spe_init controls overall).
                with torch.no_grad():
                    self.lora_spe_B.mul_(0.01)
            else:
                # gamma_spe_init == 0: keep B at zero (standard LoRA). WARNING:
                # this leaves the spectral branch dead until manually unfrozen.
                nn.init.zeros_(self.lora_spe_B)
            # FeatureSpectralGate sources from x BEFORE LoRA_B compression.
            self.feat_spectral_gate = FeatureSpectralGate(
                d_in=d_in,
                n_bands=spectral_n_bands,
                hidden_dim=spectral_hidden_dim,
                zero_init_gate=(gamma_spe_init == 0),  # see note in FeatureSpectralGate
            )
            self.gamma_spe = nn.Parameter(torch.tensor([gamma_spe_init]))

        # --- Router ---
        # Expert ordering: [sem, spa] always, +tex if TCAM, +spe if spectral.
        # n_experts:
        #   TCAM + spectral: 4 (sem/spa/tex/spe)
        #   TCAM only:       3 (sem/spa/tex)
        #   spectral only:   3 (sem/spa/spe)
        #   neither:         2 (sem/spa)
        has_tcam = self.tcam is not None
        n_experts = 2 + (1 if has_tcam else 0) + (1 if spectral_enabled else 0)
        self.router = ExpertRouter(
            d_in=d_in,
            n_experts=n_experts,
            hidden_dim=router_hidden_dim,
        )

    # ---- spatial helpers (mirror SSDLoRAModule, but both versions accessible) ----

    def _apply_spatial_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial_conv WITHOUT TCAM. Used for E_spa path."""
        B, N, r = x.shape
        n_ns = self.n_non_spatial
        non_spatial = x[:, :n_ns]
        spatial = x[:, n_ns:]
        h = w = int(math.sqrt(spatial.shape[1]))
        spatial = spatial.reshape(B, h, w, r).permute(0, 3, 1, 2)
        spatial = self.spatial_conv(spatial)
        spatial = spatial.permute(0, 2, 3, 1).reshape(B, h * w, r)
        return torch.cat([non_spatial, spatial], dim=1)

    def _apply_spatial_conv_with_tcam(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial_conv WITH TCAM. Used for E_tex path."""
        B, N, r = x.shape
        n_ns = self.n_non_spatial
        non_spatial = x[:, :n_ns]
        spatial = x[:, n_ns:]
        N_spatial = spatial.shape[1]
        side = int(math.sqrt(N_spatial))
        spatial_2d = spatial.reshape(B, side, side, r).permute(0, 3, 1, 2)
        spatial_2d = self.spatial_conv(spatial_2d)
        if self.tcam is not None:
            scale = self.tcam(spatial_2d)  # (B, r, 1, 1)
            spatial_2d = spatial_2d * scale
        spatial_out = spatial_2d.permute(0, 2, 3, 1).reshape(B, N_spatial, r)
        return torch.cat([non_spatial, spatial_out], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        result = self.frozen_linear(x)

        # Per-sample router scales (B, n_experts), all init to 1.0.
        # Ordering: [sem, spa] + [tex if tcam] + [spe if spectral].
        scales = self.router(x, self.n_non_spatial)
        s_sem = scales[:, 0]
        s_spa = scales[:, 1]
        idx = 2
        if self.tcam is not None:
            s_tex = scales[:, idx]; idx += 1
        else:
            s_tex = None
        if self.spectral_enabled:
            s_spe = scales[:, idx]; idx += 1
        else:
            s_spe = None

        # --- E_sem ---
        if self.r_sem > 0:
            sem = (x @ self.lora_sem_B.T) @ self.lora_sem_A.T
            s_sem_b = s_sem.view(B, 1, 1)
            result = result + s_sem_b * (self.alpha_sem * self._rslora_scale_sem) * sem

        # --- E_spa + (E_tex - E_spa) marginal ---
        # Both paths share lora_str_B, spatial_conv, lora_str_A. Only TCAM differs.
        # We compute spatial_conv ONCE on B_out, then derive spa_pre (no TCAM)
        # and tex_pre (with TCAM modulation) from the same spatial_conv output.
        # Then combine with per-sample scales and do a single A projection.
        if self.r_str > 0:
            B_out = x @ self.lora_str_B.T  # (B, N, r_str)

            # Project to 2D for spatial_conv. Keep non-spatial tokens aside.
            B_cur, N_cur, r_cur = B_out.shape
            n_ns = self.n_non_spatial
            non_spatial_b = B_out[:, :n_ns]
            spatial_b = B_out[:, n_ns:]
            N_spatial = spatial_b.shape[1]
            side = int(math.sqrt(N_spatial))
            spatial_2d = spatial_b.reshape(B_cur, side, side, r_cur).permute(0, 3, 1, 2)

            # spatial_conv (shared): produces the structural-only activation.
            spa_2d = self.spatial_conv(spatial_2d)  # (B, r, H, W)

            s_spa_b = s_spa.view(B, 1, 1)
            if self.tcam is not None:
                # tex marginal: spa + s_tex * (tex - spa), tex = spa * TCAM
                scale = self.tcam(spa_2d)  # (B, r, 1, 1)
                tex_2d = spa_2d * scale
                spa_pre = spa_2d.permute(0, 2, 3, 1).reshape(B_cur, N_spatial, r_cur)
                tex_pre = tex_2d.permute(0, 2, 3, 1).reshape(B_cur, N_spatial, r_cur)
                spa_pre = torch.cat([non_spatial_b, spa_pre], dim=1)  # (B, N, r)
                tex_pre = torch.cat([non_spatial_b, tex_pre], dim=1)
                s_tex_b = s_tex.view(B, 1, 1)
                combined = s_spa_b * spa_pre + s_tex_b * (tex_pre - spa_pre)
            else:
                # No tex expert; spa only.
                spa_pre = spa_2d.permute(0, 2, 3, 1).reshape(B_cur, N_spatial, r_cur)
                spa_pre = torch.cat([non_spatial_b, spa_pre], dim=1)
                combined = s_spa_b * spa_pre

            structural = combined @ self.lora_str_A.T  # (B, N, d_out)
            result = result + (self.alpha_str * self._rslora_scale_str) * structural

        # --- E_spe (high-source spectral, optional) ---
        # Always compute spe when spectral_enabled so gradient can flow back to
        # gamma_spe even when gamma_spe=0 at init (d_loss/d_gamma_spe = spe).
        # gamma_spe=0 means contribution is 0 at init, but the path must run.
        if self.spectral_enabled:
            gate = self.feat_spectral_gate(x, self.n_non_spatial)  # (B, D)
            gated_x = x * gate.unsqueeze(1)  # (B, N, D)
            spe = (gated_x @ self.lora_spe_B.T) @ self.lora_spe_A.T  # (B, N, d_out)
            s_spe_b = s_spe.view(B, 1, 1)
            result = result + self.gamma_spe * s_spe_b * self._rslora_scale_spe * spe

        return result


# ---------------------------------------------------------------------------
# Feature Fusion
# ---------------------------------------------------------------------------

class FeatureFusionModule(nn.Module):
    """Multi-layer feature extraction, channel reduction, and fusion."""

    def __init__(self, in_channels_list: List[int], embed_dim: int):
        super().__init__()
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, embed_dim, 1),
                    nn.BatchNorm2d(embed_dim),
                    nn.GELU(),
                )
                for c in in_channels_list
            ]
        )
        fused_dim = embed_dim * len(in_channels_list)
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_dim, fused_dim, 3, padding=1, groups=fused_dim, bias=False),
            nn.BatchNorm2d(fused_dim),
            nn.GELU(),
            nn.Conv2d(fused_dim, fused_dim, 1, bias=False),
            nn.BatchNorm2d(fused_dim),
            nn.GELU(),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        target_size = features[0].shape[2:]
        projected = []
        for feat, proj in zip(features, self.projections):
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            projected.append(proj(feat))
        fused = torch.cat(projected, dim=1)
        return self.fuse(fused)


# ---------------------------------------------------------------------------
# MLP Decoder (R0 baseline)
# ---------------------------------------------------------------------------

class MLPDecoder(nn.Module):
    """Lightweight MLP decoder for segmentation."""

    def __init__(self, in_channels_list: List[int], embed_dim: int = 256,
                 num_classes: int = 7):
        super().__init__()
        self.fusion = FeatureFusionModule(in_channels_list, embed_dim)
        fused_dim = embed_dim * len(in_channels_list)
        self.fused_dim = fused_dim
        self.head = nn.Sequential(
            nn.Conv2d(fused_dim, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, num_classes, 1),
        )

    def forward(self, features: List[torch.Tensor], img_size: Tuple[int, int],
                return_features: bool = False):
        # fused: (B, fused_dim, H/16, W/16) — token-resolution decoder feature,
        # the natural target for affinity preservation.
        fused = self.fusion(features)
        out = F.interpolate(fused, size=img_size, mode="bilinear", align_corners=False)
        logits = self.head(out)
        if return_features:
            return logits, fused
        return logits


# ---------------------------------------------------------------------------
# LAR-D++ Decoder (B2-main, spec 20260622_b2_main_lar_dpp_decoder_design.md)
# ---------------------------------------------------------------------------

class LARDppDecoder(nn.Module):
    """LAR-D++ = DPT-style 1+1+2+2 reassembly + FRM + PRN + head @ H/4.

    Drop-in replacement for MLPDecoder. Same input (list of 6 spatial tensors
    at H/16), different output resolution (H/4 vs MLP's H/16).

    Tap grouping (0-indexed in features list, corresponding to 1-indexed
    layers [3, 6, 9, 12, 18, 24]):
        c2 @ H/4:  features[0]                  (layer 3, early)
        c3 @ H/8:  features[1]                  (layer 6, early-mid)
        c4 @ H/16: features[2] + features[3]    (layers 9+12, mid)
        c5 @ H/32: features[4] + features[5]    (layers 18+24, late)

    Reuses MM-DINO's FRM (cross-scale reinforcement) and PRN (top-down +
    bottom-up refinement).
    """

    def __init__(self, in_channels_list: List[int], embed_dim: int = 256,
                 num_classes: int = 6, frm_out_d: Optional[int] = None,
                 prn_stages: int = 1, **unused):
        # Lazy import — keeps the import cost off the MLP-decoder path.
        from mmdino_modules import (
            FeatureReinforcementModule,
            ProgressiveRefinementNeck,
        )
        super().__init__()
        assert len(in_channels_list) == 6, (
            f"LAR-D++ expects 6 taps; got {len(in_channels_list)}. "
            f"Set cfg.layers_to_extract = [3, 6, 9, 12, 18, 24]."
        )
        # Taps paired for c4 / c5 must share channel count so the sum is valid.
        assert in_channels_list[2] == in_channels_list[3], (
            f"c4 pair taps must share channel count; got "
            f"{in_channels_list[2]} vs {in_channels_list[3]}"
        )
        assert in_channels_list[4] == in_channels_list[5], (
            f"c5 pair taps must share channel count; got "
            f"{in_channels_list[4]} vs {in_channels_list[5]}"
        )
        self.embed_dim = embed_dim
        # frm_out_d defaults to embed_dim — current B2-main design has FRM
        # output at embed_dim, but expose as kwarg so config can override if
        # a future ablation wants different FRM/PRN channel widths.
        frm_out_d = frm_out_d or embed_dim
        assert prn_stages >= 1, f"prn_stages must be >= 1, got {prn_stages}"

        # --- Reassembly: per-scale 1x1 projection to embed_dim ---
        self.proj_c2 = nn.Conv2d(in_channels_list[0], embed_dim, 1, bias=False)
        self.proj_c3 = nn.Conv2d(in_channels_list[1], embed_dim, 1, bias=False)
        # c4 / c5 conv sees the summed tensor (channels == in_channels_list[2/4])
        self.proj_c4 = nn.Conv2d(in_channels_list[2], embed_dim, 1, bias=False)
        self.proj_c5 = nn.Conv2d(in_channels_list[4], embed_dim, 1, bias=False)

        # --- Per-scale resize to target stride (H/16 input) ---
        self.resize_c2 = nn.ConvTranspose2d(
            embed_dim, embed_dim, kernel_size=4, stride=4
        )  # H/16 → H/4
        self.resize_c3 = nn.ConvTranspose2d(
            embed_dim, embed_dim, kernel_size=2, stride=2
        )  # H/16 → H/8
        # c4 stays at H/16 (no resize module)
        self.pool_c5 = nn.MaxPool2d(kernel_size=2, stride=2)  # H/16 → H/32

        # --- FRM (cross-scale reinforcement) ---
        # FRM expects in_d length-5: [check_slot, c2_d, c3_d, c4_d, c5_d].
        # The check_slot is unused in FRM's internal logic; mirror MMDINODecoder
        # convention by passing frm_out_d for that slot too.
        self.frm = FeatureReinforcementModule(
            in_d=[frm_out_d] + [frm_out_d] * 4,
            out_d=frm_out_d,
        )

        # --- PRN (top-down + bottom-up refinement) ---
        self.neck = ProgressiveRefinementNeck(
            channels_list=[frm_out_d] * 4,
            num_stages=prn_stages,
        )

        # --- Classifier head @ H/4 ---
        # Head input is PRN p2 (frm_out_d channels) — project to num_classes.
        self.head = nn.Sequential(
            nn.Conv2d(frm_out_d, frm_out_d, 3, padding=1, bias=False),
            nn.BatchNorm2d(frm_out_d),
            nn.GELU(),
            nn.Conv2d(frm_out_d, num_classes, 1),
        )

    def forward(self, features: List[torch.Tensor], img_size: Tuple[int, int],
                return_features: bool = False):
        """Decode 6 spatial tap features into per-pixel logits.

        Args:
            features: list of 6 tensors at H/16, each (B, C, H/16, W/16)
            img_size: (H, W) target image size for final logits
            return_features: if True, also return the pre-head decoder feature
                (PRN p2 output @ H/4, embed_dim channels). Required for
                consistency loss + diagnostic paths that currently call
                `self.decoder(..., return_features=True)` — see
                src/model.py HALoRASeg.forward.

        Returns:
            logits: (B, num_classes, H, W) — bilinear-upsampled from H/4
            [optional] fused: (B, embed_dim, H/4, W/4) — PRN p2 output, named
                `fused` to match MLPDecoder's return_features contract.
        """
        assert len(features) == 6, f"Expected 6 features, got {len(features)}"

        # Step 1: 1+1+2+2 DPT-style reassembly
        c2 = self.resize_c2(self.proj_c2(features[0]))                  # → H/4
        c3 = self.resize_c3(self.proj_c3(features[1]))                  # → H/8
        c4 = self.proj_c4(features[2] + features[3])                    # → H/16
        c5 = self.pool_c5(self.proj_c5(features[4] + features[5]))      # → H/32

        # Step 2: FRM (4-scale cross-reinforcement) → s2/s3/s4/s5
        s2, s3, s4, s5 = self.frm(c2, c3, c4, c5)

        # Step 3: PRN (top-down + bottom-up) → take p2 @ H/4
        p2, _, _ = self.neck([s2, s3, s4, s5])

        # Step 4: head + upsample to image resolution
        logits_h4 = self.head(p2)
        logits = F.interpolate(
            logits_h4, size=img_size, mode="bilinear", align_corners=False
        )
        if return_features:
            return logits, p2
        return logits


# ---------------------------------------------------------------------------
# MM-DINO Decoder Wrapper (P1, v7.2 plan §P1)
# ---------------------------------------------------------------------------

class MMDINODecoderFull(nn.Module):
    """HALoRASeg-compatible wrapper around MM-DINO SampleAdapter + MMDINODecoder.

    Same input contract as MLPDecoder / LARDppDecoder:
        forward(features: List[(B, C, H/16, W/16)], img_size, return_features=False)

    But expects 4 taps (not 6). SampleAdapter operates on per-tap spatial features
    directly (its internal permute/reshape is skipped — we already have spatial).

    Pipeline:
        4 taps @ H/16, embed_dim
          → SampleAdapter projects + resizes → pyramid [H/4, H/8, H/16, H/32]
                                            with channels [256, 512, 1024, 1024]
          → MMDINODecoder (FRM + PRN + out_conv) → logits @ H/4
          → bilinear upsample → (H, W)

    The decayed_lambda kwarg is propagated to FRM's FeatureFusionModule residuals
    (Paper1284 CVPR 2026 free fold-in). Default 1.0 = no-op (original MM-DINO).
    """

    def __init__(
        self,
        in_channels_list: List[int],
        embed_dim: int = 256,
        num_classes: int = 6,
        adapter_out_channels: tuple = (256, 512, 1024, 1024),
        decayed_lambda: float = 1.0,
        fad_v2_channels: int = None,
        epfo_cfg: dict = None,
        external_branch_cfg: dict = None,
        tap3_resize: str = "stride_conv",
        prn_output_stage: str = "full",
        prn_resize_mode: str = "nearest",
        prn_nonlinear: bool = False,
        decoder_variant: str = "full",
        **unused,
    ):
        from mmdino_modules import (
            SampleAdapter, MMDINODecoder, FeatureReinforcementModule,
            ProgressiveRefinementNeck, ConvBNReLU,
        )
        super().__init__()
        assert len(in_channels_list) == 4, (
            f"MMDINODecoderFull expects 4 taps (MM-DINO spec); got {len(in_channels_list)}. "
            f"Set cfg.layers_to_extract = [5, 12, 18, 24] (1-indexed config lands on "
            f"0-indexed [4, 11, 17, 23])."
        )
        self.embed_dim = embed_dim
        self._skip_adapter = decoder_variant == "linear_head_noadapter"
        # All 4 taps share the same embed_dim (ViT-L output channels). SampleAdapter
        # accepts a single `in_channels` arg — verify all taps match.
        assert all(c == in_channels_list[0] for c in in_channels_list), (
            f"All 4 taps must share in_channels for SampleAdapter; got {in_channels_list}"
        )

        # Optional FAD-v2 block on c2 (identity-safe Haar wavelet
        # residual). None = disabled (preserves A1/S1 baseline behavior).
        # See src/fad_v2.py for design rationale.
        if fad_v2_channels is not None:
            from fad_v2 import FADv2Block
            self.fad_v2 = FADv2Block(channels=fad_v2_channels, zero_init=True)
            print(
                f"[FAD-v2] enabled on c2 (channels={fad_v2_channels}, "
                f"params={sum(p.numel() for p in self.fad_v2.parameters()):,}, "
                f"identity-safe at step 0)"
            )
        else:
            self.fad_v2 = None

        # Optional EPFO sparse boundary refinement head.
        # Operates on PRN p2 @ H/4 + coarse logits_h4 (post-out_conv) to refine
        # logits at top-K uncertain boundary points via MLP. None = disabled.
        if epfo_cfg is not None and epfo_cfg.get("enabled", False):
            from epfo_head import EPFOHead
            self.epfo = EPFOHead(
                embed_dim=int(epfo_cfg.get("embed_dim", embed_dim)),
                num_classes=int(epfo_cfg.get("num_classes", num_classes)),
                num_fcs=int(epfo_cfg.get("num_fcs", 3)),
                hidden_dim=int(epfo_cfg.get("hidden_dim", 256)),
                theta=int(epfo_cfg.get("theta", 5)),
                ratio=float(epfo_cfg.get("ratio", 0.7)),
                coarse_pred_each_layer=bool(epfo_cfg.get("coarse_pred_each_layer", True)),
            )
            print(
                f"[EPFO] enabled on p2+logits_h4 (params="
                f"{sum(p.numel() for p in self.epfo.parameters()):,}, "
                f"theta={self.epfo.theta}, ratio={self.epfo.ratio})"
            )
        else:
            self.epfo = None

        # SampleAdapter: per-tap 1x1 conv (D → adapter_out_channels[i]) + resize.
        # We extract projects/resize_layers and apply directly to spatial features,
        # skipping the (B, N, D) → (B, D, H, W) permute/reshape inside SampleAdapter.forward
        # (HALoRASeg already produces spatial maps).
        if not self._skip_adapter:
            self._adapter_proj = nn.ModuleList([
                nn.Conv2d(
                    in_channels=in_channels_list[0],
                    out_channels=oc,
                    kernel_size=1, stride=1, padding=0,
                )
                for oc in adapter_out_channels
            ])
            # tap3_resize (decoder Innovation 3 Tier-1.1, 2026-08-14):
            #   "stride_conv" (default, original MM-DINO): Conv2d k=3 s=2 p=1, 9.43M params
            #   "maxpool":       MaxPool2d(2), 0 params — primary surgery
            #   "depthwise_k3_s2": depthwise Conv2d k=3 s=2 p=1 groups=C bias=False,
            #                    ~9K params — fallback
            c3 = adapter_out_channels[3]
            if tap3_resize == "stride_conv":
                tap3_layer = nn.Conv2d(c3, c3, kernel_size=3, stride=2, padding=1)
            elif tap3_resize == "maxpool":
                tap3_layer = nn.MaxPool2d(kernel_size=2, stride=2)
            elif tap3_resize == "depthwise_k3_s2":
                tap3_layer = nn.Conv2d(c3, c3, kernel_size=3, stride=2, padding=1,
                                        groups=c3, bias=False)
            else:
                raise ValueError(
                    f"Unknown tap3_resize={tap3_resize!r}; expected one of "
                    f"['stride_conv', 'maxpool', 'depthwise_k3_s2']"
                )
            self._tap3_resize = tap3_resize
            self._adapter_resize = nn.ModuleList([
                nn.ConvTranspose2d(adapter_out_channels[0], adapter_out_channels[0],
                                   kernel_size=4, stride=4, padding=0),  # H/16 → H/4
                nn.ConvTranspose2d(adapter_out_channels[1], adapter_out_channels[1],
                                   kernel_size=2, stride=2, padding=0),  # H/16 → H/8
                nn.Identity(),                                             # H/16 → H/16
                tap3_layer,                                                # H/16 → H/32
            ])
        else:
            self._adapter_proj = nn.ModuleList()
            self._adapter_resize = nn.ModuleList()
            self._tap3_resize = tap3_resize

        # MMDINODecoder: FRM + PRN + out_conv. lambda_decayed propagates to FRM.
        # prn_output_stage/prn_resize_mode (Innovation 3 Tier-1.2, 2026-08-14):
        #   "p2_only" + "unified" — drop degenerate P3/P4 output
        #   branch (−1.77M) + bilinear-up/MaxPool-down resizes. Defaults preserve
        #   the original MM-DINO behavior.
        # decoder_variant (MM-DINO style part-level ablation, 2026-08-24):
        #   full        = FRM + PRN + out_conv (our new decoder, default)
        #   frm         = FRM(+λ,tap3 improvements) + out_conv only
        #   prn         = PRN(+prnstage,prnresize improvements) + out_conv only
        #   linear_head = 1x1 conv on p1 (no FRM/PRN parts) — ablation base
        self.decoder_variant = decoder_variant
        self._skip_adapter = decoder_variant == "linear_head_noadapter"
        if decoder_variant == "full":
            self._decoder = MMDINODecoder(
                n_classes=num_classes,
                in_channels=adapter_out_channels,
                out_channels=embed_dim,
                lambda_decayed=decayed_lambda,
                prn_output_stage=prn_output_stage,
                prn_resize_mode=prn_resize_mode,
                prn_nonlinear=prn_nonlinear,
            )
            self._variant_out_conv = self._decoder.out_conv
        elif decoder_variant == "frm":
            self._frm = FeatureReinforcementModule(
                [adapter_out_channels[0]] + list(adapter_out_channels),
                out_d=embed_dim,
                lambda_decayed=decayed_lambda,
            )
            self._variant_out_conv = ConvBNReLU(embed_dim, num_classes, 1, pad=0)
        elif decoder_variant == "prn":
            self._neck = ProgressiveRefinementNeck(
                channels_list=list(adapter_out_channels),
                output_stage=prn_output_stage,
                resize_mode=prn_resize_mode,
                nonlinear=prn_nonlinear,
            )
            self._variant_out_conv = ConvBNReLU(
                adapter_out_channels[0], num_classes, 1, pad=0)
        elif decoder_variant == "linear_head":
            self._variant_out_conv = nn.Conv2d(
                adapter_out_channels[0], num_classes, kernel_size=1)
        elif decoder_variant == "linear_head_noadapter":
            # MM-DINO Table V baseline: frozen backbone last-tap feature (H/16,
            # raw channels) -> 1x1 linear head. NO SampleAdapter/FRM/PRN.
            self._variant_out_conv = nn.Conv2d(
                in_channels_list[0], num_classes, kernel_size=1)
        else:
            raise ValueError(
                f"Unknown decoder_variant={decoder_variant!r}; expected one of "
                f"['full', 'frm', 'prn', 'linear_head', 'linear_head_noadapter']"
            )
        # expose fused_dim for consistency with MLPDecoder contract (eval code reads this)
        self.fused_dim = embed_dim

        # External Spatial Pathway: raw-image conv stem (RI) or shallow-tap (ST).
        from external_branch import build_external_branch, build_layer_scale
        self.external_branch_cfg = external_branch_cfg or {}
        self.external_branch = (
            build_external_branch(self.external_branch_cfg)
            if self.external_branch_cfg.get("enabled", False) else None
        )
        self.external_branch_enabled = self.external_branch is not None
        if self.external_branch_enabled:
            self.external_gamma = build_layer_scale(self.external_branch_cfg, dim=embed_dim)
            self.external_variant = self.external_branch_cfg.get("variant", "RI").upper()
            self.external_inject_mode = self.external_branch_cfg.get(
                "inject_mode", "serial"  # Variant E sets "parallel_concat"
            )
            # Variant E parallel-concat projection — built HERE (not lazily in forward) so it
            # registers as a parameter before optimizer param groups are constructed.
            # Lazy build would skip optimizer registration → projection stays at zero-init
            # forever → Variant E silently tests "serial + zero" instead of parallel concat.
            if self.external_inject_mode == "parallel_concat":
                self._parallel_concat_proj = nn.Conv2d(
                    embed_dim * 2, embed_dim, kernel_size=1, bias=True
                )
                nn.init.zeros_(self._parallel_concat_proj.weight)
                nn.init.zeros_(self._parallel_concat_proj.bias)
            print(
                f"[External Pathway] enabled: variant={self.external_variant}, "
                f"inject_mode={self.external_inject_mode}, "
                f"gamma={'frozen' if not self.external_gamma.gamma.requires_grad else 'trainable'}"
            )
        else:
            self.external_gamma = None
            self.external_variant = None
            self.external_inject_mode = "serial"

    def forward(self, features: List[torch.Tensor], img_size: Tuple[int, int],
                return_features: bool = False,
                return_p2: bool = False,
                raw_image: torch.Tensor = None,
                shallow_tap: torch.Tensor = None):
        assert len(features) == 4, f"Expected 4 features, got {len(features)}"

        # Step 1: per-tap project + resize → pyramid [c2@H/4, c3@H/8, c4@H/16, c5@H/32]
        if not self._skip_adapter:
            c2 = self._adapter_resize[0](self._adapter_proj[0](features[0]))
            c3 = self._adapter_resize[1](self._adapter_proj[1](features[1]))
            c4 = self._adapter_proj[2](features[2])  # Identity resize
            c4 = self._adapter_resize[2](c4)
            c5 = self._adapter_resize[3](self._adapter_proj[3](features[3]))
        else:
            c2 = c3 = c4 = c5 = None

        # Step 1.5: optional FAD-v2 on c2 (identity-safe Haar residual).
        # Between SampleAdapter output and FRM input. No-op when self.fad_v2 is None.
        if self.fad_v2 is not None:
            c2 = self.fad_v2(c2)

        # Step 2: decoder → logits @ H/4 (variant-aware, MM-DINO part ablation).
        if self.decoder_variant == "full":
            s2, s3, s4, s5 = self._decoder.frm(c2, c3, c4, c5)
            p2 = self._decoder.neck([s2, s3, s4, s5])[0]
        elif self.decoder_variant == "frm":
            s2, s3, s4, s5 = self._frm(c2, c3, c4, c5)
            p2 = s2
        elif self.decoder_variant == "prn":
            p2 = self._neck([c2, c3, c4, c5])[0]
        elif self.decoder_variant == "linear_head":
            p2 = c2
        else:  # linear_head_noadapter: raw last-tap backbone feature @ H/16
            p2 = features[3]

        # Step 2.5a: External Spatial Pathway injection.
        # Variant RI: branch(raw_image) at H/4; serial residual: p2 = p2 + γ ⊙ branch(x).
        # Variant ST: branch(shallow_tap) at H/4; same serial residual.
        # Variant E (parallel concat): handled here — concatenate along channel
        # dim then 1×1 conv project back. Ablation only.
        # No-op when external_branch is None (Variant B / disabled).
        if self.external_branch_enabled:
            if self.external_variant == "RI":
                if raw_image is None:
                    raise ValueError(
                        "External pathway variant=RI requires raw_image kwarg; "
                        "got None. HALoRASeg.forward must pass raw_image=x to decoder."
                    )
                branch_out = self.external_branch(raw_image)
            elif self.external_variant == "ST":
                if shallow_tap is None:
                    raise ValueError(
                        "External pathway variant=ST requires shallow_tap kwarg; "
                        "got None. HALoRASeg.forward must compute backbone layer-1 tap "
                        "and pass it down."
                    )
                branch_out = self.external_branch(shallow_tap)
            else:
                raise ValueError(f"Unknown external_variant: {self.external_variant!r}")

            if self.external_inject_mode == "serial":
                # Houlsby #2: serial residual (positive result)
                p2 = p2 + self.external_gamma(branch_out)
            elif self.external_inject_mode == "parallel_concat":
                # Variant E ablation: parallel concat (Houlsby negative result test).
                # _parallel_concat_proj built in __init__ (zero-init weight + bias)
                # so it registers in optimizer param groups.
                concat = torch.cat([p2, branch_out], dim=1)
                p2 = p2 + self.external_gamma(self._parallel_concat_proj(concat))
            else:
                raise ValueError(f"Unknown inject_mode: {self.external_inject_mode!r}")

        logits_h4 = self._variant_out_conv(p2)            # (B, num_classes, H/4, W/4)

        # Step 2.5: optional EPFO sparse boundary refinement.
        # Refines logits_h4 at top-K uncertain boundary points via MLP([p2, probs]).
        # No-op (and p2 unused) when self.epfo is None.
        if self.epfo is not None:
            logits_h4 = self.epfo(p2, logits_h4)

        # Step 3: bilinear upsample to img_size
        logits = F.interpolate(
            logits_h4, size=img_size, mode="bilinear", align_corners=False
        )

        if return_p2:
            # Step 1.25 router audit: expose pre-out_conv PRN output
            # (p2 @ H/4, embed_dim channels). Distinct from return_features which
            # returns logits_h4 (num_classes channels) for legacy consistency loss.
            return logits, logits_h4, p2
        if return_features:
            # Legacy consistency-loss path: returns logits_h4 (num_classes chans).
            return logits, logits_h4
        return logits


# ---------------------------------------------------------------------------
# GEAS Segmentation Model
# ---------------------------------------------------------------------------

class HALoRASeg(nn.Module):
    """GEAS segmentation model."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        # --- Backbone ---
        backbone_cfg = cfg.get("backbone", {})
        backbone_name = backbone_cfg.get("name", "vits16")
        checkpoint = backbone_cfg.get("checkpoint", None)
        self.backbone = _build_dinov3_backbone(backbone_name, checkpoint)
        self.embed_dim = self.backbone.embed_dim
        self.depth = len(self.backbone.blocks)
        self.n_storage_tokens = self.backbone.n_storage_tokens
        self.patch_size = self.backbone.patch_size

        # Freeze all backbone params first; SSD-LoRA params are added after
        for param in self.backbone.parameters():
            param.requires_grad = False

        # --- SSD-LoRA ---
        lora_cfg = cfg.get("lora", {})
        # lora_blocks: 1-indexed in config → 0-indexed internally
        if "lora_blocks" in lora_cfg and lora_cfg["lora_blocks"] is not None:
            self._lora_blocks_0idx = [b - 1 for b in lora_cfg["lora_blocks"]]
        else:
            self._lora_blocks_0idx = None
        # mode="none" = paper-faithful base MM-DINO: backbone stays fully frozen,
        # no LoRA injected, decoder-only training (used for MM-DINO paper reproduction).
        if lora_cfg.get("mode", "ssd") == "none":
            self._lora_blocks_0idx = None
            print("[LoRA] mode=none — backbone fully frozen, no LoRA injected")
        else:
            self._apply_ssd_lora(lora_cfg)

        # --- Feature extraction layers (1-indexed in config, 0-indexed internally) ---
        decoder_type = cfg.get("decoder", {}).get("type", "mlp")
        if "layers_to_extract" in cfg:
            self.layers_to_extract = cfg["layers_to_extract"]
        else:
            self.layers_to_extract = [3, 6, 9, 12]
        self.layers_to_extract_0idx = [l - 1 for l in self.layers_to_extract]

        # --- Decoder ---
        decoder_cfg = cfg.get("decoder", {})
        decoder_embed_dim = decoder_cfg.get("embed_dim", 96 if self.embed_dim < 512 else 256)
        num_classes = cfg.get("num_classes", 7)
        in_channels = [self.embed_dim] * len(self.layers_to_extract)

        if decoder_type == "mlp":
            self.decoder = MLPDecoder(in_channels, decoder_embed_dim, num_classes)
        elif decoder_type == "lar_dpp":
            # B2-main LAR-D++ (spec 20260622_b2_main_lar_dpp_decoder_design.md)
            # Requires 6 taps [3, 6, 9, 12, 18, 24]; enforced inside LARDppDecoder.
            # frm_out_d and prn_stages are config knobs (decoder.frm_out_d /
            # decoder.prn_stages); both default to embed_dim / 1 inside the
            # decoder if absent.
            self.decoder = LARDppDecoder(
                in_channels,
                decoder_embed_dim,
                num_classes,
                frm_out_d=decoder_cfg.get("frm_out_d", None),
                prn_stages=decoder_cfg.get("prn_stages", 1),
            )
        elif decoder_type == "mmdino":
            # P1 (v7.2 plan): MM-DINO byte-port — SampleAdapter + FRM + PRN + out_conv.
            # Requires 4 taps at 0-indexed [4, 11, 17, 23] (MM-DINO spec). Since this
            # codebase's config is 1-indexed, write layers_to_extract=[5,12,18,24].
            # lambda_decayed propagates to FRM residual (Paper1284 CVPR 2026 free fold-in).
            # fad_v2: optional identity-safe Haar wavelet
            # residual block on c2. Enabled via decoder.fad_v2.enabled in config.
            fad_v2_cfg = decoder_cfg.get("fad_v2", {}) or {}
            fad_v2_enabled = bool(fad_v2_cfg.get("enabled", False))
            fad_v2_channels = None
            if fad_v2_enabled:
                # Default to adapter_out_channels[0] (256 for MM-DINO c2).
                fad_v2_channels = int(fad_v2_cfg.get(
                    "channels",
                    decoder_cfg.get("adapter_out_channels", (256, 512, 1024, 1024))[0],
                ))
            # EPFO: sparse boundary refinement head on
            # p2 + logits_h4. Enabled via decoder.epfo.enabled in config.
            epfo_cfg = decoder_cfg.get("epfo", {}) or {}
            # External Spatial Pathway: raw-image conv stem (primary) or
            # shallow-tap (ablation), serial-injected into p2 with zero-init
            # γ LayerScale.
            external_branch_cfg = decoder_cfg.get("external_branch", {}) or None
            self.decoder = MMDINODecoderFull(
                in_channels,
                decoder_embed_dim,
                num_classes,
                adapter_out_channels=tuple(decoder_cfg.get(
                    "adapter_out_channels", (256, 512, 1024, 1024)
                )),
                decayed_lambda=float(decoder_cfg.get("decayed_lambda", 1.0)),
                fad_v2_channels=fad_v2_channels,
                epfo_cfg=epfo_cfg if epfo_cfg.get("enabled", False) else None,
                external_branch_cfg=external_branch_cfg if (
                    external_branch_cfg and external_branch_cfg.get("enabled", False)
                ) else None,
                tap3_resize=str(decoder_cfg.get("tap3_resize", "stride_conv")),
                prn_output_stage=str(decoder_cfg.get("prn_output_stage", "full")),
                prn_resize_mode=str(decoder_cfg.get("prn_resize_mode", "nearest")),
                prn_nonlinear=bool(decoder_cfg.get("prn_nonlinear", False)),
            )
        elif decoder_type == "ha_mmdino":
            # HA-Decoder v1 (paper_verified.md, 2026-06-26):
            #   FRM (Paper1284 decayed, reused) + GPR (ChangeDINO FuseGated)
            #   + PFEB (Lian 2026) + out_conv.
            # Same 4-tap contract as MMDINODecoderFull. Warm-starts from P1
            # `mmdino` checkpoint via shared SampleAdapter + FRM parameter names.
            # New modules (GPR/PFEB) are identity-init'd so the initial forward
            # matches P1's output — see src/ha_decoder.py docstring.
            from ha_decoder import HADecoderFull
            self.decoder = HADecoderFull(
                in_channels,
                decoder_embed_dim,
                num_classes,
                adapter_out_channels=tuple(decoder_cfg.get(
                    "adapter_out_channels", (256, 512, 1024, 1024)
                )),
                decayed_lambda=float(decoder_cfg.get("decayed_lambda", 1.0)),
                use_gpr=bool(decoder_cfg.get("use_gpr", True)),
                use_pfeb=bool(decoder_cfg.get("use_pfeb", True)),
                pfeb_expansion=int(decoder_cfg.get("pfeb_expansion", 2)),
            )
        elif decoder_type == "wpd":
            # WPD Decoder (V11 B1, 2026-06-29): Wavelet-Prototype Decoder.
            #   WPD-FRM (WavFFM replaces FeatureFusionModule — Haar wavelet gating)
            #   + P5-PPM (UPerNet-style pyramid pooling on P5 for global context)
            #   + standard PRN + out_conv.
            # Same 4-tap contract as MMDINODecoderFull / HADecoderFull. Warm-starts
            # from HA-Decoder v1 / P1 checkpoints (SampleAdapter + conv_fuse + conv_identity
            # share parameter names; only spatial_attn + p5_ppm are new).
            # See src/wpd_decoder.py docstring for design + identity-init contract.
            from wpd_decoder import WPDDecoderFull
            self.decoder = WPDDecoderFull(
                in_channels,
                decoder_embed_dim,
                num_classes,
                adapter_out_channels=tuple(decoder_cfg.get(
                    "adapter_out_channels", (256, 512, 1024, 1024)
                )),
                decayed_lambda=float(decoder_cfg.get("decayed_lambda", 1.0)),
                gate_bias=float(decoder_cfg.get("wavffm_gate_bias", 1.0)),
            )
        else:
            raise ValueError(
                f"Unknown decoder type: {decoder_type}. "
                f"Supported: 'mlp', 'lar_dpp', "
                f"'mmdino' (full MM-DINO), 'ha_mmdino' (HA-Decoder v1), "
                f"'wpd' (Wavelet-Prototype Decoder). "
                f"Historical configs (pfu/sru/upernet/mask2former/sc_cmrd_lar) "
                f"are abandoned and no longer buildable."
            )

        # --- Optional Late SF-Adapter ---
        # Installed via forward hooks on backbone.blocks[indices]. Disabled by
        # default; enabled only for R1 experiments. See src/sf_adapter.py.
        sf_cfg = cfg.get("sf_adapter", {}) or {}
        self._sf_enabled = bool(sf_cfg.get("enabled", False))
        if self._sf_enabled:
            from sf_adapter import install_sf_adapters

            # blocks are 1-indexed in config (e.g. [19, 20, 21, 22, 23, 24])
            blocks_1idx = sf_cfg.get("blocks", [19, 20, 21, 22, 23, 24])
            blocks_0idx = [b - 1 for b in blocks_1idx]
            in_ch = int(sf_cfg.get("in_channels", self.embed_dim))
            bottleneck = int(sf_cfg.get("bottleneck", 256))
            n_bands = int(sf_cfg.get("n_bands", 8))
            handles = install_sf_adapters(
                self,
                block_indices_0idx=blocks_0idx,
                in_channels=in_ch,
                bottleneck=bottleneck,
                n_bands=n_bands,
            )
            self._sf_block_indices_0idx = blocks_0idx
            self._sf_hook_handles = handles
            n_sf = sum(p.numel() for p in self.sf_adapters.parameters())
            print(
                f"[SF-Adapter] installed on blocks {blocks_1idx} (0-idx {blocks_0idx}): "
                f"in_ch={in_ch}, bottleneck={bottleneck}, n_bands={n_bands}, "
                f"params={n_sf:,}"
            )

        # --- Optional Earth-Adapter E1 ---
        # DINOv3-safe hook port: preserves CLS + storage tokens, applies
        # raw/FFT-routed adapter only to spatial patch tokens.
        earth_cfg = cfg.get("earth_adapter", {}) or {}
        self._earth_enabled = bool(earth_cfg.get("enabled", False))
        if self._earth_enabled:
            from earth_adapter import install_earth_adapters

            blocks_1idx = earth_cfg.get("blocks", list(range(1, self.depth + 1)))
            fft_layers_1idx = earth_cfg.get(
                "fft_layers",
                earth_cfg.get("fft_layer", []),
            )
            blocks_0idx = [int(b) - 1 for b in blocks_1idx]
            fft_layers_0idx = [int(b) - 1 for b in fft_layers_1idx]
            bottleneck = int(earth_cfg.get("bottleneck", earth_cfg.get("dim", 64)))
            cutoff_ratio = float(earth_cfg.get("cutoff_ratio", 0.3))
            scale = float(earth_cfg.get("scale", 0.1))
            handles = install_earth_adapters(
                self,
                block_indices_0idx=blocks_0idx,
                fft_layers_0idx=fft_layers_0idx,
                channels=self.embed_dim,
                bottleneck=bottleneck,
                cutoff_ratio=cutoff_ratio,
                scale=scale,
            )
            self._earth_hook_handles = handles
            n_earth = sum(p.numel() for p in self.earth_adapters.parameters())
            print(
                f"[Earth-Adapter] installed on blocks {blocks_1idx} "
                f"(fft_layers={fft_layers_1idx}): bottleneck={bottleneck}, "
                f"cutoff_ratio={cutoff_ratio}, scale={scale}, params={n_earth:,}"
            )

    # ---- rank helpers ----

    def _get_rank_for_block(self, block_idx: int) -> Tuple[int, int]:
        if self._lora_blocks_0idx is not None and block_idx not in self._lora_blocks_0idx:
            return (0, 0)

        lora_cfg = self.cfg.get("lora", {})

        rank_config = lora_cfg.get("rank_config", "progressive")
        semantic_only = lora_cfg.get("semantic_only", False)

        if rank_config == "progressive":
            progressive_ranks = self.cfg.get("lora", {}).get("progressive_ranks", None)
            if progressive_ranks:
                n_segments = len(progressive_ranks)
                segment_size = max(1, self.depth // n_segments)
                seg_idx = min(block_idx // segment_size, n_segments - 1)
                r = progressive_ranks[seg_idx]
                return (r, 0) if semantic_only else (r // 2, r // 2)
            if self.depth <= 12:
                if block_idx < 4: r = 4
                elif block_idx < 8: r = 8
                else: r = 16
            else:  # 24-block ViT-L
                if block_idx < 8: r = 4
                elif block_idx < 16: r = 8
                else: r = 16
            return (r, 0) if semantic_only else (r // 2, r // 2)
        elif rank_config == "progressive_split":
            ranks_sem = lora_cfg.get("progressive_ranks_sem", None)
            ranks_str = lora_cfg.get("progressive_ranks_str", None)
            if not ranks_sem or not ranks_str:
                raise ValueError(
                    "rank_config='progressive_split' requires both "
                    "lora.progressive_ranks_sem and lora.progressive_ranks_str"
                )
            if len(ranks_sem) != len(ranks_str):
                raise ValueError(
                    "progressive_ranks_sem and progressive_ranks_str must have "
                    f"the same length, got {len(ranks_sem)} and {len(ranks_str)}"
                )
            n_segments = len(ranks_sem)
            segment_size = max(1, self.depth // n_segments)
            seg_idx = min(block_idx // segment_size, n_segments - 1)
            r_sem = int(ranks_sem[seg_idx])
            r_str = int(ranks_str[seg_idx])
            return (r_sem, 0) if semantic_only else (r_sem, r_str)
        elif rank_config == "uniform":
            r = self.cfg.get("lora", {}).get("uniform_rank", 8)
            return (r, 0) if semantic_only else (r // 2, r // 2)
        else:
            raise ValueError(f"Unknown rank config: {rank_config}")

    # ---- SSD-LoRA application ----

    def _apply_ssd_lora(self, lora_cfg: dict):
        target_modules = lora_cfg.get("target_modules", ["qkv", "fc1", "fc2"])
        lora_mode = lora_cfg.get("mode", "ssd")
        structural_path_type = lora_cfg.get("structural_path_type", "single")
        n_non_spatial = 1 + self.n_storage_tokens
        alpha_init = lora_cfg.get("alpha_init", 1.0)
        use_rslora = lora_cfg.get("use_rslora", False)

        # TCAM config (Phase 3b-S)
        # If structural_path_type starts with "tcam_", enter TCAM mode.
        # tcam_base_path controls the spatial processor under TCAM (single,
        # multi_scale_dwconv, or conv_lora_moe for Phase 1 Conv-LoRA).
        # tcam_blocks (1-indexed in config) restricts TCAM to a subset.
        if structural_path_type.startswith("tcam_"):
            tcam_type = structural_path_type
            base_structural_path = lora_cfg.get("tcam_base_path", "single")
            if base_structural_path not in ("single", "multi_scale_dwconv", "conv_lora_moe"):
                raise ValueError(
                    f"tcam_base_path must be 'single', 'multi_scale_dwconv', "
                    f"or 'conv_lora_moe', got: {base_structural_path}"
                )
        else:
            tcam_type = None
            base_structural_path = structural_path_type

        # Conv-LoRA MoE knobs (Phase 1). Read unconditionally; only used when
        # base_structural_path == "conv_lora_moe".
        conv_lora_n_experts = int(lora_cfg.get("conv_lora_n_experts", 8))
        conv_lora_top_k = int(lora_cfg.get("conv_lora_top_k", 1))
        conv_lora_upsample_ratios = lora_cfg.get("conv_lora_upsample_ratios", None)
        conv_lora_loss_coef = float(lora_cfg.get("conv_lora_loss_coef", 1e-2))

        tcam_gamma = lora_cfg.get("tcam_gamma", 0.2)
        tcam_hidden_min = lora_cfg.get("tcam_hidden_min", 8)

        # tcam_blocks: 1-indexed in config -> 0-indexed internally
        tcam_blocks_cfg = lora_cfg.get("tcam_blocks", None)
        if tcam_blocks_cfg is not None:
            tcam_blocks_0idx = sorted(set(int(b) - 1 for b in tcam_blocks_cfg))
            # validate range
            invalid = [b + 1 for b in tcam_blocks_0idx if not (0 <= b < self.depth)]
            if invalid:
                raise ValueError(
                    f"tcam_blocks contains invalid 1-indexed block numbers: {invalid}. "
                    f"Valid range: 1..{self.depth}"
                )
        else:
            tcam_blocks_0idx = None  # means all blocks (only meaningful if tcam_type set)

        # --- Run A-main: Routed SF-SSD-LoRA (override 2026-06-17) ---
        # When mode == "routed_sf_ssd", we wrap each linear in RoutedSSDLoRAModule
        # instead of SSDLoRAModule. Routed requires TCAM (tex expert = TCAM marginal).
        # Spectral is optional (spectral_cfg.enabled); defaults off in Run A-main.
        is_routed = (lora_mode == "routed_sf_ssd")
        spectral_cfg = lora_cfg.get("spectral", {}) or {}
        spectral_enabled = bool(spectral_cfg.get("enabled", False))
        r_spe = int(spectral_cfg.get("r_spe", 0)) if spectral_enabled else 0
        spectral_n_bands = int(spectral_cfg.get("n_bands", 4))
        spectral_hidden_dim = int(spectral_cfg.get("hidden_dim", 64))
        gamma_spe_init = float(spectral_cfg.get("gamma_spe_init", 1e-3))
        router_hidden_dim = int(lora_cfg.get("router_hidden_dim", 64))

        if is_routed:
            # Routed mode routes between {sem, spa, [tex if TCAM], [spe if spectral]}.
            # At least one of TCAM or spectral must be configured.
            if tcam_type is None and not spectral_enabled:
                raise ValueError(
                    "lora.mode='routed_sf_ssd' requires either "
                    "structural_path_type starting with 'tcam_' (tex expert) "
                    "or spectral.enabled=true (spe expert)."
                )
            if spectral_enabled and r_spe <= 0:
                raise ValueError(
                    "spectral.enabled=true requires spectral.r_spe > 0 (high-source rank)."
                )

        # Parse spectral.ranges — per-block gamma_spe_init for differential amplitude.
        # Format: list of {blocks: [start, end], gamma_spe_init: float} (1-indexed inclusive).
        # If omitted while spectral_enabled=true, all routed blocks get spectral
        # with the global spectral.gamma_spe_init.
        spectral_ranges = None  # None means "all routed blocks"
        if spectral_enabled:
            ranges_cfg = spectral_cfg.get("ranges", None)
            if ranges_cfg is not None:
                spectral_ranges = []
                for rng in ranges_cfg:
                    blocks_spec = rng["blocks"]
                    if len(blocks_spec) != 2:
                        raise ValueError(
                            f"spectral.ranges[].blocks must be [start, end], got {blocks_spec}"
                        )
                    start, end = int(blocks_spec[0]), int(blocks_spec[1])
                    if not (1 <= start <= end <= self.depth):
                        raise ValueError(
                            f"spectral.ranges[].blocks invalid: [{start},{end}] "
                            f"for depth={self.depth}"
                        )
                    gamma = float(rng.get("gamma_spe_init", gamma_spe_init))
                    spectral_ranges.append((start, end, gamma))

        def _block_spectral_cfg(idx_0: int):
            """Returns (enabled, gamma_spe_init) for block at 0-indexed idx_0."""
            if not spectral_enabled:
                return False, 0.0
            if spectral_ranges is None:
                return True, gamma_spe_init
            idx_1 = idx_0 + 1
            for start, end, gamma in spectral_ranges:
                if start <= idx_1 <= end:
                    return True, gamma
            return False, 0.0

        n_tcam_modules = 0
        n_routed_modules = 0
        n_spectral_modules = 0
        for i, block in enumerate(self.backbone.blocks):
            r_sem, r_str = self._get_rank_for_block(i)
            if r_sem == 0 and r_str == 0:
                continue

            # Decide per-block TCAM
            if (tcam_type is not None and
                (tcam_blocks_0idx is None or i in tcam_blocks_0idx)):
                block_tcam_type = tcam_type
                n_tcam_modules += 1
            else:
                block_tcam_type = None

            # Decide per-block spectral
            block_spe_enabled, block_gamma_spe = _block_spectral_cfg(i)

            if is_routed:
                # A block is routed if it has TCAM OR spectral enabled.
                # Otherwise (no TCAM, no spectral) fall back to vanilla SSD-LoRA.
                is_block_routed = (block_tcam_type is not None) or block_spe_enabled
                if is_block_routed:
                    common_kwargs = dict(
                        r_sem=r_sem, r_str=r_str, n_non_spatial=n_non_spatial,
                        structural_path_type=base_structural_path,
                        use_rslora=use_rslora, tcam_type=block_tcam_type,
                        tcam_gamma=tcam_gamma, tcam_hidden_min=tcam_hidden_min,
                        spectral_enabled=block_spe_enabled,
                        r_spe=r_spe if block_spe_enabled else 0,
                        spectral_n_bands=spectral_n_bands,
                        spectral_hidden_dim=spectral_hidden_dim,
                        gamma_spe_init=block_gamma_spe,
                        router_hidden_dim=router_hidden_dim,
                    )
                    ModuleCls = RoutedSSDLoRAModule
                    n_routed_modules += 1
                    if block_spe_enabled:
                        n_spectral_modules += 1
                else:
                    common_kwargs = dict(
                        r_sem=r_sem, r_str=r_str, n_non_spatial=n_non_spatial,
                        lora_mode="ssd", structural_path_type=base_structural_path,
                        use_rslora=use_rslora, tcam_type=None,
                        tcam_gamma=tcam_gamma, tcam_hidden_min=tcam_hidden_min,
                        conv_lora_n_experts=conv_lora_n_experts,
                        conv_lora_top_k=conv_lora_top_k,
                        conv_lora_upsample_ratios=conv_lora_upsample_ratios,
                        conv_lora_loss_coef=conv_lora_loss_coef,
                    )
                    ModuleCls = SSDLoRAModule
            else:
                common_kwargs = dict(
                    r_sem=r_sem, r_str=r_str, n_non_spatial=n_non_spatial,
                    lora_mode=lora_mode, structural_path_type=base_structural_path,
                    use_rslora=use_rslora, tcam_type=block_tcam_type,
                    tcam_gamma=tcam_gamma, tcam_hidden_min=tcam_hidden_min,
                    conv_lora_n_experts=conv_lora_n_experts,
                    conv_lora_top_k=conv_lora_top_k,
                    conv_lora_upsample_ratios=conv_lora_upsample_ratios,
                    conv_lora_loss_coef=conv_lora_loss_coef,
                )
                ModuleCls = SSDLoRAModule

            if "qkv" in target_modules:
                block.attn.qkv = ModuleCls(block.attn.qkv, **common_kwargs)
            if "fc1" in target_modules:
                block.mlp.fc1 = ModuleCls(block.mlp.fc1, **common_kwargs)
            if "fc2" in target_modules:
                block.mlp.fc2 = ModuleCls(block.mlp.fc2, **common_kwargs)

        # Print TCAM summary for traceability
        if tcam_type is not None:
            if tcam_blocks_0idx is not None and len(tcam_blocks_0idx) > 1:
                tcam_blocks_print = f"[{tcam_blocks_0idx[0]+1}..{tcam_blocks_0idx[-1]+1}]"
            elif tcam_blocks_0idx is not None:
                tcam_blocks_print = f"{[b+1 for b in tcam_blocks_0idx]}"
            else:
                tcam_blocks_print = "all"
            print(
                f"TCAM: type={tcam_type}, blocks_1idx={tcam_blocks_print}, "
                f"base_path={base_structural_path}, gamma={tcam_gamma}, "
                f"hidden_min={tcam_hidden_min}, "
                f"tcam_modules={n_tcam_modules*len(target_modules)} "
                f"({n_tcam_modules} blocks x {len(target_modules)} targets)"
            )

        if is_routed:
            spe_str = (
                f", spectral_blocks={n_spectral_modules}/{n_routed_modules}, "
                f"r_spe={r_spe}, n_bands={spectral_n_bands}"
                if n_spectral_modules > 0 else ", spectral=off"
            )
            print(
                f"Routed SF-SSD-LoRA: routed_modules={n_routed_modules*len(target_modules)} "
                f"({n_routed_modules} blocks x {len(target_modules)} targets), "
                f"router_hidden_dim={router_hidden_dim}"
                + spe_str
            )

        # Initialize alpha values if specified
        if alpha_init != 1.0:
            for block in self.backbone.blocks:
                for attr in ["attn.qkv", "mlp.fc1", "mlp.fc2"]:
                    module = attr.split(".")
                    mod = block
                    for m in module:
                        mod = getattr(mod, m)
                    if isinstance(mod, SSDLoRAModule):
                        if mod.r_sem > 0:
                            mod.alpha_sem.data.fill_(alpha_init)
                        if mod.r_str > 0:
                            mod.alpha_str.data.fill_(alpha_init)
                    elif isinstance(mod, RoutedSSDLoRAModule):
                        if mod.r_sem > 0:
                            mod.alpha_sem.data.fill_(alpha_init)
                        if mod.r_str > 0:
                            mod.alpha_str.data.fill_(alpha_init)

        # alpha_trainable=false: freeze alpha scalars (standard-LoRA baseline,
        # Hu et al. 2021 set alpha to the first r tried and do not tune it).
        if not lora_cfg.get("alpha_trainable", True):
            for block in self.backbone.blocks:
                for attr in ["attn.qkv", "mlp.fc1", "mlp.fc2"]:
                    mod = block
                    for m in attr.split("."):
                        mod = getattr(mod, m)
                    if isinstance(mod, (SSDLoRAModule, RoutedSSDLoRAModule)):
                        if mod.r_sem > 0:
                            mod.alpha_sem.requires_grad_(False)
                        if mod.r_str > 0:
                            mod.alpha_str.requires_grad_(False)
            print("SSD-LoRA alpha: FROZEN (alpha_trainable=false; standard-LoRA baseline)")

        trainable = count_trainable_parameters(self)
        if self._lora_blocks_0idx is not None:
            n_lora = len(self._lora_blocks_0idx)
            blocks_str = f"blocks {self._lora_blocks_0idx} ({n_lora}/{len(self.backbone.blocks)})"
        else:
            blocks_str = f"all {len(self.backbone.blocks)} blocks"
        rslora_str = " (rsLoRA: alpha/sqrt(r))" if use_rslora else ""
        print(
            f"SSD-LoRA (mode={lora_mode}) applied to {blocks_str}, "
            f"targets={target_modules}, alpha_init={alpha_init}{rslora_str}, "
            f"trainable params={trainable:,}"
        )

    # ---- forward ----

    def get_aux_losses(self) -> dict:
        """Aggregate auxiliary losses from leaf submodules.

        Returns a dict of {name: scalar_tensor}. Callers (e.g. train_cross.py)
        add these to the main loss with their own coefficients.

        Currently harvested:
          - ``moe_loss``: sum of last_moe_loss across all ConvLoRAMoE leaves.
            Already scaled by each module's ``loss_coef`` (default 1e-2 per
            AutoGluon code-faithful port). Zero when no Conv-LoRA MoE modules
            exist or before the first forward.

        Implementation notes:
          - Device: uses ``next(self.parameters()).device`` so the returned
            tensor sits on the model's actual device (CPU/GPU), avoiding
            device-mismatch when callers add it to a GPU loss tensor.
          - Double-count avoidance: iterates ``isinstance(mod, ConvLoRAMoE)``
            rather than ``hasattr(mod, "last_moe_loss")`` so a wrapper module
            that aliases the same tensor (e.g. SSDLoRAModule mirroring its
            child ConvLoRAMoE) cannot cause the same loss to count twice.
        """
        from conv_lora import ConvLoRAMoE  # lazy import avoids circular
        device = next(self.parameters()).device
        moe_total = torch.zeros(1, device=device)
        for mod in self.modules():
            if isinstance(mod, ConvLoRAMoE):
                moe_total = moe_total + mod.last_moe_loss
        return {"moe_loss": moe_total.squeeze()}

    def forward(self, x: torch.Tensor, return_features: bool = False,
                return_p2: bool = False):
        B, C, H, W = x.shape
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        features_raw = self.backbone.get_intermediate_layers(
            x, n=self.layers_to_extract_0idx, reshape=False, norm=True
        )

        feature_maps = []
        for feat in features_raw:
            # feat: (B, N_patches, C) — CLS and storage tokens already stripped
            feat = feat.reshape(B, h_patches, w_patches, -1).permute(0, 3, 1, 2).contiguous()
            feature_maps.append(feat)

        # External pathway ST variant: extract DINOv3 layer-1 tap (H/16 grid, un-reshape).
        # Only computed when decoder asks for it (variant=ST). RI uses raw_image directly.
        shallow_tap = None
        if (hasattr(self.decoder, "external_variant") and
                self.decoder.external_variant == "ST"):
            shallow_tap = self.backbone.get_intermediate_layers(
                x, n=[1], reshape=False, norm=True, return_class_token=False,
            )[0]  # (B, N=(H/16)², D=1024)

        # Pass raw_image=x so external pathway (variant RI) can compute branch(raw_image).
        # No-op when decoder has no external_branch (default path) — extra kwargs ignored.
        if return_p2:
            logits, logits_h4, p2 = self.decoder(
                feature_maps, img_size=(H, W), return_p2=True,
                raw_image=x, shallow_tap=shallow_tap,
            )
            return logits, logits_h4, p2
        if return_features:
            logits, fused = self.decoder(
                feature_maps, img_size=(H, W), return_features=True,
                raw_image=x, shallow_tap=shallow_tap,
            )
            return logits, fused
        return self.decoder(
            feature_maps, img_size=(H, W),
            raw_image=x, shallow_tap=shallow_tap,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_model(cfg) -> HALoRASeg:
    """Factory function: create GEAS model from config dict."""
    return HALoRASeg(cfg)


def count_trainable_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
