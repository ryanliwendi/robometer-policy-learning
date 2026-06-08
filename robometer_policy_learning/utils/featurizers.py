import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from typing import Dict, List, Optional, Union

try:
    # Reuse helper from transformer utilities to identify image-like keys
    from robometer_policy_learning.modules.transformer.transformer_utils import identify_image_keys
except Exception:

    def identify_image_keys(obs_keys: List[str]) -> List[str]:
        image_keywords = ["image", "rgb", "camera", "vision", "visual"]
        return [k for k in obs_keys if any(s in k.lower() for s in image_keywords)]


def _build_mlp_layers(input_size, hidden_dims, activation, use_layer_norm=False, dropout_rate=0.0):
    """Build a list of MLP layers (used by featurizers and by the MLP actor/critic)."""
    layers = []
    prev_size = int(input_size)
    for hidden_size in hidden_dims:
        hidden_size = int(hidden_size)
        layers.append(nn.Linear(prev_size, hidden_size))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_size))
        if activation.lower() == "relu":
            layers.append(nn.ReLU())
        elif activation.lower() == "tanh":
            layers.append(nn.Tanh())
        elif activation.lower() == "elu":
            layers.append(nn.ELU())
        elif activation.lower() == "leaky_relu":
            layers.append(nn.LeakyReLU())
        else:
            raise ValueError(f"Unknown activation: {activation}")
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))
        prev_size = hidden_size
    return layers


class ObservationFeaturizer(nn.Module):
    """Per-key observation featurizer for MLP-based actors/critics.

    Image keys are encoded by featurizer-level image encoders (impala | resnet | dinov2,
    built via ``modules.encoders.build_image_featurizers``); the remaining low-dim/vector
    keys go through per-key MLP encoders. The per-key features are concatenated (in the
    order of ``featurizer_cfg``) into a single ``(B, output_dim)`` vector.

    Mirrors the structure of ``TransformerFeatureExtractor`` (``image_encoders`` +
    ``lowdim_encoders``) so both feature paths share the same vocabulary.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        featurizer_cfg: Optional[Dict[str, Union[List, nn.Module]]] = None,
        activation: str = "relu",
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
        # Image encoder parameters (optional). image_encoder_type in {impala, resnet, dinov2}
        # enables featurizer-level encoding of image keys; None falls back to MLP/flatten.
        image_encoder_type: Optional[str] = None,
        finetune_image_encoder: bool = False,
        image_feature_dim: int = 128,
        # ResNet
        resnet_backbone: str = "ResNet18",
        resnet_pretrained: bool = True,
        resnet_pool: str = "spatial_softmax",
        spatial_softmax_num_kp: int = 32,
        # DINOv2 (model/processor required when image_encoder_type == "dinov2")
        dinov2_model: object = None,
        dinov2_processor: object = None,
        # IMPALA
        impala_nn_scale: int = 1,
        impala_num_blocks_per_stack: int = 2,
        impala_use_smaller: bool = False,
        impala_output_dim: int = None,
    ):
        super().__init__()
        self.observation_space = observation_space
        featurizer_cfg = dict(featurizer_cfg or {})
        is_dict = isinstance(observation_space, gym.spaces.Dict)

        # --- Image encoders (only for image keys, only when an image encoder is requested) ---
        self.image_keys: List[str] = []
        if image_encoder_type in ("impala", "resnet", "dinov2"):
            if not is_dict:
                raise ValueError("Image encoders require a Dict observation space")
            self.image_keys = identify_image_keys(list(observation_space.spaces.keys()))

        self.image_encoders = nn.ModuleDict()
        if self.image_keys:
            from robometer_policy_learning.modules.encoders import build_image_featurizers

            self.image_encoders.update(
                build_image_featurizers(
                    observation_space=observation_space,
                    image_keys=self.image_keys,
                    image_encoder_type=image_encoder_type,
                    finetune=finetune_image_encoder,
                    output_dim=impala_output_dim,
                    image_feature_dim=image_feature_dim,
                    resnet_backbone=resnet_backbone,
                    resnet_pretrained=resnet_pretrained,
                    resnet_pool=resnet_pool,
                    spatial_softmax_num_kp=spatial_softmax_num_kp,
                    impala_nn_scale=impala_nn_scale,
                    impala_num_blocks_per_stack=impala_num_blocks_per_stack,
                    impala_use_smaller=impala_use_smaller,
                    dinov2_model=dinov2_model,
                    dinov2_processor=dinov2_processor,
                )
            )

        # Make sure image keys are accounted for in the concat order even if the caller did
        # not list them in featurizer_cfg.
        for k in self.image_keys:
            featurizer_cfg.setdefault(k, None)

        # --- Low-dim MLP encoders for the remaining configured keys ---
        # _out_dims maps each key to its post-encoding feature size (for analytic output_dim).
        self.lowdim_encoders = nn.ModuleDict()
        self._out_dims: Dict[str, int] = {}
        for key, value in featurizer_cfg.items():
            if key in self.image_encoders:
                self._out_dims[key] = self.image_encoders[key].output_dim
                continue
            if isinstance(value, (list, tuple)):
                in_dim = int(np.prod(observation_space.spaces[key].shape))
                self.lowdim_encoders[key] = nn.Sequential(
                    *_build_mlp_layers(in_dim, value, activation, use_layer_norm, dropout_rate)
                )
                self._out_dims[key] = int(value[-1])
            elif isinstance(value, nn.Module):
                self.lowdim_encoders[key] = value
                od = getattr(value, "output_dim", None)
                if od is None:  # infer from a 1-sample forward (cheap for an MLP)
                    in_dim = int(np.prod(observation_space.spaces[key].shape))
                    with torch.no_grad():
                        od = int(value(torch.zeros(1, in_dim)).shape[-1])
                self._out_dims[key] = int(od)
            else:
                raise ValueError(f"Featurizer for key {key} must be list/tuple or nn.Module, got {type(value)}")

        # Ordered keys to read from obs and concatenate.
        if featurizer_cfg:
            self._keys: Optional[List[str]] = list(featurizer_cfg.keys())
        elif is_dict:
            self._keys = [k for k, s in observation_space.spaces.items() if getattr(s, "shape", None) is not None]
        else:
            self._keys = None  # plain-tensor observation

        self._output_dim = self._compute_output_dim()

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def _compute_output_dim(self) -> int:
        """Sum the per-key feature sizes (encoded dim, or raw flattened dim if un-encoded)."""
        if self._keys is None:
            return int(np.prod(self.observation_space.shape))
        total = 0
        for k in self._keys:
            dim = self._out_dims.get(k)
            if dim is None:  # un-encoded key -> raw flatten
                dim = int(np.prod(self.observation_space.spaces[k].shape))
            total += int(dim)
        return total

    @staticmethod
    def _vec(v: torch.Tensor) -> torch.Tensor:
        """Flatten a per-key value to (B, D)."""
        return v.reshape(v.size(0), -1) if v.dim() > 1 else v.unsqueeze(0)

    def _flatten_tensor(self, obs: torch.Tensor, device: Optional[torch.device]) -> torch.Tensor:
        if obs.dim() > 2:
            result = obs.reshape(obs.size(0), -1)
        elif obs.dim() == 1:
            result = obs.unsqueeze(0)
        elif obs.dim() == 2 and obs.size(1) == 1 and obs.size(0) != 1:
            result = obs.t()  # (features, 1) -> (1, features)
        else:
            result = obs
        return result.to(device) if device is not None else result

    def flatten_obs(self, obs: Union[dict, torch.Tensor], device: Optional[torch.device] = None) -> torch.Tensor:
        """Encode and concatenate observations into a single ``(B, output_dim)`` tensor."""
        if not isinstance(obs, dict):
            return self._flatten_tensor(obs, device)

        feats = []
        for k in self._keys or []:
            if k not in obs:
                continue  # key configured but absent at runtime -> skip
            v = obs[k]
            if device is not None:
                v = v.to(device)
            if k in self.image_encoders:
                feats.append(self.image_encoders[k](v))  # encoder handles dtype/shape/normalization
            elif k in self.lowdim_encoders:
                feats.append(self.lowdim_encoders[k](self._vec(v).float()))
            else:
                feats.append(self._vec(v).float())  # un-encoded key -> raw flatten

        if not feats:
            raise ValueError(
                f"No valid features found in observation dict. "
                f"Keys present: {list(obs.keys())}, expected keys: {self._keys}"
            )
        return torch.cat(feats, dim=-1)
