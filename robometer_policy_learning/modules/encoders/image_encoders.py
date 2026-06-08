"""
Pluggable image-observation encoders used at the featurizer level.

This module provides a single, reusable set of image featurizers (IMPALA, ResNet,
DINOv2) behind a common interface so the MLP, transformer, and RNN actors/critics can
share the same encoding code instead of each reimplementing it.

Contract for every featurizer here:
  * ``forward(x)`` accepts images shaped ``(B, H, W, C)`` / ``(B, C, H, W)`` (and a
    leading singleton frame dim ``(B, 1, H, W, C)``), as uint8 ``[0, 255]`` or float
    ``[0, 1]``, normalizes internally, and returns ``(B, output_dim)``.
  * ``.output_dim`` reports the feature dimensionality.
  * ``finetune`` controls whether the encoder's parameters are trainable. When
    ``finetune=False`` the encoder is frozen and kept in ``eval()`` (so BatchNorm /
    dropout running stats don't drift), and its forward runs under ``torch.no_grad()``.

Use :func:`build_image_featurizer` (single key) or :func:`build_image_featurizers`
(per-key over an observation space) to construct these.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from robometer_policy_learning.modules.encoders.impala_encoder import ImpalaEncoder, SmallerImpalaEncoder


def _to_bchw_float(x: torch.Tensor) -> torch.Tensor:
    """Normalize an arbitrary image tensor to ``(B, C, H, W)`` float in ``[0, 1]``.

    Accepts ``(H,W,C)``/``(C,H,W)`` (adds batch), ``(B,1,H,W,C)`` (drops the frame dim),
    ``(B,H,W,C)`` and ``(B,C,H,W)``. Values in ``[0, 255]`` are scaled to ``[0, 1]``.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor image, got {type(x)}")

    # Drop a leading singleton frame/stack dim: (B, 1, H, W, C) -> (B, H, W, C)
    if x.dim() == 5 and x.size(1) == 1:
        x = x.squeeze(1)

    if x.dim() == 3:
        x = x.unsqueeze(0)

    if x.dim() != 4:
        raise ValueError(f"Expected 3D/4D image tensor, got shape {tuple(x.shape)}")

    # Channels-last -> channels-first
    if x.size(-1) in (1, 3) and x.size(1) not in (1, 3):
        x = x.permute(0, 3, 1, 2).contiguous()

    x = x.float()
    # Scale to [0, 1] if it looks like a [0, 255] image
    if x.max() > 1.0:
        x = x / 255.0
    return x


class ImpalaImageFeaturizer(nn.Module):
    """Image featurizer backed by the IMPALA CNN encoder (trained from scratch)."""

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        nn_scale: int = 1,
        num_blocks_per_stack: int = 2,
        use_smaller: bool = False,
        output_dim: Optional[int] = None,
        finetune: bool = True,
    ):
        super().__init__()
        if use_smaller:
            self.encoder = SmallerImpalaEncoder(
                input_shape=input_shape, nn_scale=nn_scale, output_dim=output_dim
            )
        else:
            self.encoder = ImpalaEncoder(
                input_shape=input_shape,
                nn_scale=nn_scale,
                num_blocks_per_stack=num_blocks_per_stack,
                output_dim=output_dim,
            )

        self.finetune = finetune
        for p in self.encoder.parameters():
            p.requires_grad = finetune
        if not finetune:
            self.encoder.eval()

    @property
    def output_dim(self) -> int:
        return self.encoder.output_dim

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.finetune:
            self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_bchw_float(x.to(next(self.parameters()).device))
        # ImpalaEncoder handles its own normalization, but we already produced [0,1] floats.
        if self.finetune:
            return self.encoder(x)
        with torch.no_grad():
            return self.encoder(x)


class SpatialSoftmax(nn.Module):
    """Spatial Softmax pooling as used in robomimic (expected keypoint coordinates)."""

    def __init__(self, input_shape: Tuple[int, int, int], num_kp: int = 32, temperature: float = 1.0):
        super().__init__()
        self.input_shape = input_shape  # (C, H, W)
        self.num_kp = num_kp
        self.temperature = temperature

        C, H, W = input_shape
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, W), np.linspace(-1.0, 1.0, H))
        self.register_buffer("pos_x", torch.from_numpy(pos_x.reshape(H * W)).float())
        self.register_buffer("pos_y", torch.from_numpy(pos_y.reshape(H * W)).float())
        self.fc = nn.Linear(C, num_kp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert (C, H, W) == self.input_shape, f"{(C, H, W)} != {self.input_shape}"
        x_flat = x.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
        attention = F.softmax(self.fc(x_flat) / self.temperature, dim=1)  # (B, H*W, num_kp)
        expected_x = torch.sum(attention * self.pos_x.unsqueeze(-1), dim=1)  # (B, num_kp)
        expected_y = torch.sum(attention * self.pos_y.unsqueeze(-1), dim=1)  # (B, num_kp)
        return torch.cat([expected_x, expected_y], dim=1)  # (B, num_kp * 2)


class ResNetImageFeaturizer(nn.Module):
    """ResNet-backbone image featurizer (robomimic-style VisualCore).

    ``finetune`` is decoupled from ``resnet_pretrained``: you can load pretrained weights
    and still keep them frozen (default), or finetune from scratch / from pretrained.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        image_feature_dim: int = 128,
        resnet_backbone: str = "ResNet18",
        resnet_pretrained: bool = True,
        resnet_pool: str = "spatial_softmax",
        spatial_softmax_num_kp: int = 32,
        finetune: bool = False,
    ):
        super().__init__()
        try:
            import torchvision

            torchvision.disable_beta_transforms_warning()
            import torchvision.models as models
        except ImportError as e:
            raise ImportError("torchvision is required for the ResNet image encoder") from e

        # Normalize input shape to (C, H, W)
        input_shape = tuple(input_shape)
        if len(input_shape) == 4 and input_shape[0] == 1:
            input_shape = input_shape[1:]
        if len(input_shape) == 3 and input_shape[-1] in (1, 3) and input_shape[0] not in (1, 3):
            input_shape = (input_shape[2], input_shape[0], input_shape[1])  # HWC -> CHW
        self.input_shape = input_shape

        backbones = {
            "ResNet18": models.resnet18,
            "ResNet34": models.resnet34,
            "ResNet50": models.resnet50,
        }
        if resnet_backbone not in backbones:
            raise ValueError(f"Unsupported resnet_backbone: {resnet_backbone}")
        backbone = backbones[resnet_backbone](weights="DEFAULT" if resnet_pretrained else None)

        if self.input_shape[0] != 3:
            backbone.conv1 = nn.Conv2d(self.input_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Drop avgpool + fc -> spatial feature map
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        self.finetune = finetune
        if not finetune:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()  # keep BN running stats fixed

        # ImageNet normalization (applied to [0,1] inputs)
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_shape)
            feat_shape = self.backbone(dummy).shape[1:]  # (C, H, W)

        if resnet_pool == "spatial_softmax":
            self.pool = SpatialSoftmax(input_shape=feat_shape, num_kp=spatial_softmax_num_kp)
            pool_out = spatial_softmax_num_kp * 2
        elif resnet_pool == "adaptive_avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            pool_out = int(feat_shape[0])
        elif resnet_pool == "flatten":
            self.pool = nn.Flatten()
            pool_out = int(np.prod(feat_shape))
        else:
            raise ValueError(f"Unsupported resnet_pool: {resnet_pool}")

        self.projection = nn.Linear(pool_out, image_feature_dim)
        self.output_dim = int(image_feature_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.finetune:
            self.backbone.eval()
        return self

    def _run_backbone(self, x: torch.Tensor) -> torch.Tensor:
        if self.finetune:
            return self.backbone(x)
        with torch.no_grad():
            return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_bchw_float(x.to(self.img_mean.device))
        if self.input_shape[0] == 3:
            x = (x - self.img_mean) / self.img_std
        features = self._run_backbone(x)
        pooled = self.pool(features)
        if pooled.dim() > 2:
            pooled = pooled.flatten(1)
        return self.projection(pooled)


def _dino_preprocess_params(processor) -> Tuple[List[float], List[float], int]:
    """Extract ImageNet mean/std and target square size from a HF image processor."""
    mean = list(getattr(processor, "image_mean", [0.485, 0.456, 0.406])) if processor is not None else [0.485, 0.456, 0.406]
    std = list(getattr(processor, "image_std", [0.229, 0.224, 0.225])) if processor is not None else [0.229, 0.224, 0.225]
    size = 224
    if processor is not None:
        crop = getattr(processor, "crop_size", None)
        if isinstance(crop, dict) and "height" in crop:
            size = int(crop["height"])
        else:
            sz = getattr(processor, "size", None)
            if isinstance(sz, dict):
                size = int(sz.get("shortest_edge", sz.get("height", 224)))
    return mean, std, size


class DinoImageFeaturizer(nn.Module):
    """DINOv2 image featurizer with differentiable, GPU-side preprocessing.

    Unlike a frozen precompute pipeline (HF processor on CPU + ``no_grad``), this resizes
    and normalizes on-device with differentiable ops so the backbone can be finetuned
    end-to-end. With ``finetune=False`` it behaves like a frozen encoder (eval + no_grad).
    """

    def __init__(
        self,
        dinov2_model: Any,
        dinov2_processor: Any = None,
        image_feature_dim: Optional[int] = None,
        finetune: bool = False,
    ):
        super().__init__()
        # Allow passing a model id string (load it), or a ready nn.Module.
        if isinstance(dinov2_model, str):
            from transformers import AutoImageProcessor, AutoModel

            if dinov2_processor is None or isinstance(dinov2_processor, str):
                dinov2_processor = AutoImageProcessor.from_pretrained(dinov2_processor or dinov2_model)
            dinov2_model = AutoModel.from_pretrained(dinov2_model)

        self.dino = dinov2_model
        self.finetune = finetune

        mean, std, size = _dino_preprocess_params(dinov2_processor)
        self.target_size = size
        self.register_buffer("img_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor(std).view(1, 3, 1, 1))

        for p in self.dino.parameters():
            p.requires_grad = finetune
        if not finetune:
            self.dino.eval()

        hidden = int(getattr(self.dino.config, "hidden_size", 768))
        if image_feature_dim is not None:
            self.projection = nn.Linear(hidden, int(image_feature_dim))
            self.output_dim = int(image_feature_dim)
        else:
            self.projection = None
            self.output_dim = hidden

    @property
    def device(self):
        return self.img_mean.device

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.finetune:
            self.dino.eval()
        return self

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_bchw_float(x.to(self.device))
        x = F.interpolate(x, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
        return (x - self.img_mean) / self.img_std

    def _encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.dino(pixel_values=pixel_values)
        emb = getattr(outputs, "pooler_output", None)
        if emb is None:
            # Fall back to the CLS token of the last hidden state.
            emb = outputs.last_hidden_state[:, 0]
        return emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pixel_values = self._preprocess(x)
        if self.finetune:
            emb = self._encode(pixel_values)
        else:
            with torch.no_grad():
                emb = self._encode(pixel_values)
        emb = emb.to(dtype=torch.float32)
        if self.projection is not None:
            emb = self.projection(emb)
        return emb


def build_image_featurizer(
    image_encoder_type: str,
    input_shape: Tuple[int, ...],
    *,
    finetune: bool = False,
    output_dim: Optional[int] = None,
    image_feature_dim: int = 128,
    # ResNet
    resnet_backbone: str = "ResNet18",
    resnet_pretrained: bool = True,
    resnet_pool: str = "spatial_softmax",
    spatial_softmax_num_kp: int = 32,
    # IMPALA
    impala_nn_scale: int = 1,
    impala_num_blocks_per_stack: int = 2,
    impala_use_smaller: bool = False,
    # DINOv2
    dinov2_model: Any = None,
    dinov2_processor: Any = None,
) -> nn.Module:
    """Construct a single image featurizer for one image key.

    Args:
        image_encoder_type: one of ``"impala"``, ``"resnet"``, ``"dinov2"``.
        input_shape: image observation shape (H, W, C) / (C, H, W); leading singleton dims ok.
        finetune: whether the encoder's parameters are trainable.
        output_dim: optional projection dim for IMPALA (passed to its encoder).
    Returns:
        An ``nn.Module`` exposing ``.output_dim``.
    """
    etype = (image_encoder_type or "").lower()
    if etype == "impala":
        return ImpalaImageFeaturizer(
            input_shape=input_shape,
            nn_scale=impala_nn_scale,
            num_blocks_per_stack=impala_num_blocks_per_stack,
            use_smaller=impala_use_smaller,
            output_dim=output_dim,
            finetune=finetune,
        )
    if etype == "resnet":
        return ResNetImageFeaturizer(
            input_shape=input_shape,
            image_feature_dim=image_feature_dim,
            resnet_backbone=resnet_backbone,
            resnet_pretrained=resnet_pretrained,
            resnet_pool=resnet_pool,
            spatial_softmax_num_kp=spatial_softmax_num_kp,
            finetune=finetune,
        )
    if etype == "dinov2":
        if dinov2_model is None:
            raise ValueError("image_encoder_type='dinov2' requires a dinov2_model (nn.Module or model id).")
        return DinoImageFeaturizer(
            dinov2_model=dinov2_model,
            dinov2_processor=dinov2_processor,
            image_feature_dim=output_dim,  # optional projection
            finetune=finetune,
        )
    raise ValueError(f"Unknown image_encoder_type: {image_encoder_type!r} (expected impala|resnet|dinov2)")


def build_image_featurizers(
    observation_space,
    image_keys: Optional[List[str]] = None,
    *,
    image_encoder_type: str = "impala",
    **kwargs,
) -> Dict[str, nn.Module]:
    """Build a ``{image_key: featurizer}`` dict over an observation space.

    Auto-detects image keys when ``image_keys`` is None. The per-key ``input_shape`` is read
    from the observation space. Extra kwargs are forwarded to :func:`build_image_featurizer`.
    """
    import gymnasium as gym

    # Local import avoids a hard dependency cycle at module import time.
    from robometer_policy_learning.modules.transformer.transformer_utils import identify_image_keys

    if hasattr(observation_space, "spaces"):
        obs_keys = list(observation_space.spaces.keys())
    else:
        obs_keys = ["obs"]
    keys = image_keys if image_keys is not None else identify_image_keys(obs_keys)

    featurizers: Dict[str, nn.Module] = {}
    for k in keys:
        if isinstance(observation_space, gym.spaces.Dict):
            input_shape = observation_space.spaces[k].shape
        else:
            input_shape = observation_space.shape
        featurizers[k] = build_image_featurizer(image_encoder_type, input_shape, **kwargs)
    return featurizers
