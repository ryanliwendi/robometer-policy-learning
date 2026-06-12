import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Union, Optional
import gymnasium as gym
from collections import OrderedDict
from loguru import logger
import inspect

# ResNet/SpatialSoftmax encoders now live in modules.encoders. Re-exported here for
# backward compatibility with any code importing them from transformer_utils.
from robometer_policy_learning.modules.encoders.image_encoders import (  # noqa: E402
    ResNetImageFeaturizer as ResNetEncoder,
    SpatialSoftmax,
    build_image_featurizers,
)


class MiniLMLangEncoder:
    """Language encoder using MiniLM model for generating embeddings."""

    def __init__(self, device="cpu", model_name="sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for language encoding. "
                "Please install it with: pip install sentence-transformers"
            )

        self.device = device
        self.model = SentenceTransformer(model_name)
        self.model.to(device)

        # Provide .encode for compatibility with some callers
        self.encode = self.model.encode

    def get_lang_emb(self, lang_strings):
        """Get language embeddings for a list of language strings."""
        if isinstance(lang_strings, str):
            lang_strings = [lang_strings]

        embeddings = self.model.encode(lang_strings, convert_to_tensor=True, device=self.device)
        return embeddings


def _build_mlp_layers(input_size, hidden_dims, activation, use_layer_norm=False, dropout_rate=0.0):
    """Build MLP layers (shared utility for transformer modules)."""
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
        elif activation.lower() == "gelu":
            layers.append(nn.GELU())
        else:
            raise ValueError(f"Unknown activation: {activation}")
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))
        prev_size = hidden_size
    return nn.Sequential(*layers)


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create a causal mask for transformer attention."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.masked_fill(mask == 1, float("-inf"))


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model) for proper broadcasting
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        # pe shape: (1, max_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]  # Broadcast (1, seq_len, d_model) with (batch_size, seq_len, d_model)
        return self.dropout(x)


def flatten_observations(
    obs: Union[dict, torch.Tensor],
    featurizers: nn.ModuleDict = None,
    featurizer_cfg: dict = None,
    flatten_keys: List[str] = None,
    observation_space: gym.Space = None,
) -> torch.Tensor:
    """
    Flatten observations (shared logic for transformer modules).

    Args:
        obs: Observations (dict or tensor)
        featurizers: ModuleDict of featurizers for dict observations
        featurizer_cfg: Configuration for featurizers
        flatten_keys: Keys to flatten for dict observations
        observation_space: Observation space for shape inference

    Returns:
        Flattened observations
    """
    if isinstance(obs, dict):
        if featurizer_cfg and featurizers:
            feats = []
            for k, v in obs.items():
                if k in featurizers:
                    v_flat = v.view(v.size(0), -1) if v.dim() > 1 else v.unsqueeze(0)
                    feats.append(featurizers[k](v_flat))
                else:
                    v_flat = v.view(v.size(0), -1) if v.dim() > 1 else v.unsqueeze(0)
                    feats.append(v_flat)
            if feats:
                return torch.cat(feats, dim=-1)
            else:
                raise ValueError(f"No valid features found in observation dict. Keys present: {list(obs.keys())}")
        elif flatten_keys is not None and len(flatten_keys) > 0:
            feats = []
            batch_size = None
            for k in flatten_keys:
                v = obs.get(k, None)
                if observation_space and k in observation_space.spaces:
                    shape = observation_space.spaces[k].shape
                    flat_dim = int(np.prod(shape))
                else:
                    flat_dim = 1  # fallback

                if v is None:
                    if batch_size is None:
                        for vv in obs.values():
                            if vv is not None:
                                batch_size = vv.size(0) if vv.dim() > 1 else 1
                                break
                        if batch_size is None:
                            batch_size = 1
                    # Create appropriate device tensor
                    device = next(iter([vv.device for vv in obs.values() if vv is not None]))
                    feats.append(torch.zeros(batch_size, flat_dim, device=device))
                else:
                    v_flat = v.view(v.size(0), -1) if v.dim() > 1 else v.unsqueeze(0)
                    feats.append(v_flat)
            return torch.cat(feats, dim=-1) if feats else torch.empty(0)
        else:
            feats = [v.view(v.size(0), -1) if v.dim() > 1 else v.unsqueeze(0) for v in obs.values() if v is not None]
            return torch.cat(feats, dim=-1) if feats else torch.empty(0)
    else:
        if obs.dim() > 2:
            return obs.view(obs.size(0), -1)
        elif obs.dim() == 1:
            return obs.unsqueeze(0)
        return obs


def build_featurizers(
    featurizer_cfg: dict,
    observation_space: gym.Space,
    activation: str,
    use_layer_norm: bool = False,
    dropout_rate: float = 0.0,
) -> nn.ModuleDict:
    """Build featurizers for dict observations."""
    featurizers = nn.ModuleDict()

    for key, value in featurizer_cfg.items():
        if isinstance(value, (list, tuple)):
            if isinstance(observation_space, gym.spaces.Dict) and key in observation_space.spaces:
                obs_dim = int(np.prod(observation_space.spaces[key].shape))
            else:
                raise ValueError(f"Cannot determine observation dimension for key {key}")
            featurizers[key] = _build_mlp_layers(obs_dim, value, activation, use_layer_norm, dropout_rate)
        elif isinstance(value, nn.Module):
            featurizers[key] = value
        else:
            raise ValueError(f"Featurizer for key {key} must be list/tuple or nn.Module")

    return featurizers


def identify_image_keys(obs_keys: List[str]) -> List[str]:
    """Identify which observation keys are likely to be images."""
    image_keywords = ["image", "rgb", "camera", "vision", "visual"]
    image_keys = []

    for key in obs_keys:
        if any(keyword in key.lower() for keyword in image_keywords):
            image_keys.append(key)

    return image_keys


class TransformerFeatureExtractor(nn.Module):
    """Enhanced feature extraction module for transformer models with proper image handling."""

    def __init__(
        self,
        observation_space: gym.Space,
        featurizer_cfg: dict = None,
        activation: str = "relu",
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
        preprocess_obs_transform: List[Any] = None,
        # Image encoder parameters
        image_encoder_type: str = "resnet",  # "resnet", "dinov2", "impala", or "flatten"
        finetune_image_encoder: bool = False,  # whether image encoder params are trainable
        resnet_backbone: str = "ResNet18",  # "ResNet18", "ResNet34", "ResNet50"
        resnet_pretrained: bool = True,
        resnet_pool: str = "spatial_softmax",  # "spatial_softmax", "adaptive_avg", "flatten"
        image_feature_dim: int = 128,
        spatial_softmax_num_kp: int = 32,
        # DINOv2 encoder parameters (used when image_encoder_type == "dinov2")
        dinov2_model: Any = None,
        dinov2_processor: Any = None,
        # IMPALA encoder parameters (used when image_encoder_type == "impala")
        impala_nn_scale: int = 1,
        impala_num_blocks_per_stack: int = 2,
        impala_use_smaller: bool = False,
        impala_output_dim: int = None,
        # Language embedding parameters
        use_language_embeddings: bool = True,
        lang_embedding_dim: int = 384,  # MiniLM-L6 embedding dimension
        lang_embedding_device: str = "cpu",
    ):
        super().__init__()
        self.observation_space = observation_space
        self.featurizer_cfg = copy.deepcopy(featurizer_cfg) or {}
        self.activation = activation
        self.use_layer_norm = use_layer_norm
        self.dropout_rate = dropout_rate
        self.preprocess_obs_transform = preprocess_obs_transform or []

        # Language embedding parameters
        self.use_language_embeddings = use_language_embeddings
        self.lang_embedding_dim = lang_embedding_dim

        # Language embedding cache for efficiency
        self.lang_embedding_cache = {}

        # Initialize language encoder if needed
        if self.use_language_embeddings:
            self.lang_encoder = MiniLMLangEncoder(device=lang_embedding_device)
            print(f"Initialized MiniLM language encoder on device: {lang_embedding_device}")
        else:
            self.lang_encoder = None

        # Image encoding parameters
        self.image_encoder_type = image_encoder_type
        self.finetune_image_encoder = finetune_image_encoder
        self.resnet_backbone = resnet_backbone
        self.resnet_pretrained = resnet_pretrained
        self.resnet_pool = resnet_pool
        self.image_feature_dim = image_feature_dim
        self.spatial_softmax_num_kp = spatial_softmax_num_kp
        self.dinov2_model = dinov2_model
        self.dinov2_processor = dinov2_processor
        # Try to infer DINO feature dim; fall back to image_feature_dim if unknown
        self.dinov2_feature_dim = (
            getattr(dinov2_model.config, "hidden_size", image_feature_dim)
            if dinov2_model is not None
            else image_feature_dim
        )
        # IMPALA encoder parameters
        self.impala_nn_scale = impala_nn_scale
        self.impala_num_blocks_per_stack = impala_num_blocks_per_stack
        self.impala_use_smaller = impala_use_smaller

        # Identify image keys for special processing
        if isinstance(observation_space, gym.spaces.Dict):
            self.obs_keys = list(observation_space.spaces.keys())
        else:
            self.obs_keys = ["obs"]

        self.image_keys = identify_image_keys(self.obs_keys)

        # Add missing attributes expected by forward method
        self._expected_keys = set(self.obs_keys) if isinstance(observation_space, gym.spaces.Dict) else None
        self._processed_keys = None  # Actually processed keys
        self._key_mismatch_warned = False  # To print warning only once
        self.keys_to_ignore = set()  # Keys to ignore during processing

        # For compatibility with forward method
        self._flatten_keys = self.obs_keys if isinstance(observation_space, gym.spaces.Dict) else None

        # Build per-image-key encoders via the shared factory (impala | resnet | dinov2).
        # Stored in an nn.ModuleDict so params register, move with .to(), and can be finetuned.
        self.image_encoders = None
        if self.image_encoder_type in ("impala", "resnet", "dinov2") and self.image_keys:
            encoders = build_image_featurizers(
                observation_space,
                image_keys=self.image_keys,
                image_encoder_type=self.image_encoder_type,
                finetune=self.finetune_image_encoder,
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
            self.image_encoders = nn.ModuleDict(encoders)

        # Register image encoders in featurizer_cfg so obs_dim accounts for their output_dim.
        if self.image_encoders:
            for key, value in self.image_encoders.items():
                self.featurizer_cfg[key] = value

        # Per-key MLP encoders for the NON-image (low-dim/vector) modalities. Image keys are
        # handled by self.image_encoders in forward (avoids double-registering the encoder).
        non_image_cfg = {k: v for k, v in self.featurizer_cfg.items() if not (self.image_encoders and k in self.image_encoders)}
        self.lowdim_encoders = build_featurizers(
            non_image_cfg,
            observation_space,
            activation,
            use_layer_norm,
            dropout_rate,
        )

        # Per-modality features (each already encoded to a fixed size) are concatenated and
        # returned directly. The transformer actor/critic projects this to d_model via its own
        # obs_projection, so no extra fusion MLP is needed here.
        self.obs_dim = self._calculate_obs_dim()
        self.output_dim = self.obs_dim

    @property
    def device(self):
        """Get the current device of the module."""
        return next(self.parameters()).device

    def _calculate_obs_dim(self):
        """Calculate the expected observation dimension based on observation space."""
        if not self.featurizer_cfg:  # if we have no featurizer config, we just concatenate all the keys
            if not isinstance(self.observation_space, gym.spaces.Dict):
                if hasattr(self.observation_space, "shape") and self.observation_space.shape is not None:
                    return int(np.prod(self.observation_space.shape))
                else:
                    total_dim = 0

                    for k in self.obs_keys:
                        if k in self.observation_space.spaces:
                            shape = self.observation_space.spaces[k].shape
                            dim = int(np.prod(shape))
                            total_dim += dim

                    return total_dim if total_dim > 0 else None
        else:
            # image encoders expose .output_dim; MLP featurizers (list of dims) output their last dim
            return sum(
                v.output_dim if hasattr(v, "output_dim") else int(v[-1])
                for v in self.featurizer_cfg.values()
            )

    def _encode_image(self, key: str, v: torch.Tensor) -> torch.Tensor:
        """Encode an image observation for ``key`` via its featurizer-level encoder.

        All encoder types (impala/resnet/dinov2) share one interface: they accept raw
        images ((B,H,W,C)/(B,C,H,W), uint8 or float), normalize internally, and return
        (B, output_dim). Falls back to normalize+flatten if no encoder is configured.
        """
        if self.image_encoders is not None and key in self.image_encoders:
            return self.image_encoders[key](v)
        # Fallback: no encoder for this image key -> normalize and flatten.
        if v.max() > 1.0:
            v = v / 255.0
        return v.reshape(v.size(0), -1) if v.dim() > 1 else v.unsqueeze(0)

    def _encode_lowdim(self, key: str, feature: torch.Tensor) -> torch.Tensor:
        """Apply the per-key low-dim MLP encoder if one exists; else return the feature as-is."""
        if key in self.lowdim_encoders:
            return self.lowdim_encoders[key](feature)
        return feature

    def _encode_language_if_needed(self, language_instructions):
        """
        Encode language instructions on-demand with caching.

        Args:
            language_instructions: List of language instruction strings or None

        Returns:
            torch.Tensor: Language embeddings or None if no instructions
        """
        if not self.use_language_embeddings or not language_instructions:
            return None

        # Convert single string to list
        if isinstance(language_instructions, str):
            language_instructions = [language_instructions]

        # Check cache for existing embeddings
        embeddings = []
        uncached_instructions = []
        uncached_indices = []

        for i, instruction in enumerate(language_instructions):
            if instruction in self.lang_embedding_cache:
                embeddings.append(self.lang_embedding_cache[instruction])
            else:
                embeddings.append(None)  # Placeholder
                uncached_instructions.append(instruction)
                uncached_indices.append(i)

        # Compute embeddings for uncached instructions
        if uncached_instructions:
            with torch.no_grad():
                new_embeddings = self.lang_encoder.get_lang_emb(uncached_instructions)
                # Cache new embeddings
                for instruction, embedding in zip(uncached_instructions, new_embeddings):
                    cached_embedding = embedding.cpu()
                    self.lang_embedding_cache[instruction] = cached_embedding

                # Fill in the placeholders
                for idx, cache_idx in enumerate(uncached_indices):
                    embeddings[cache_idx] = new_embeddings[idx].cpu()

        # Convert to tensor and move to the correct device
        embeddings_tensor = torch.stack([emb.to(self.device) for emb in embeddings])
        return embeddings_tensor

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the feature extractor with consistent key ordering.

        Args:
            obs: Dictionary of observations

        Returns:
            Flattened feature tensor
        """
        if not isinstance(obs, dict):
            raise ValueError("obs must be a dictionary")

        # Get keys in consistent sorted order for reproducible behavior
        input_keys = set(obs.keys())

        # Check for key mismatches and warn if this is the first time
        if self._expected_keys is not None:
            missing_keys = self._expected_keys - input_keys
            extra_keys = input_keys - self._expected_keys

            if (missing_keys or extra_keys) and not self._key_mismatch_warned:
                # Try to infer calling stage from call stack
                try:
                    frames = inspect.stack()
                    funcs = [f.function for f in frames[:10]]
                    stage = "unknown"
                    if any("run_eval" in fn for fn in funcs):
                        stage = "EVAL"
                    elif any("run_rollout" in fn for fn in funcs):
                        stage = "ROLLOUT"
                    elif any("train_loop" in fn or "run_learner" in fn for fn in funcs):
                        stage = "TRAIN"
                except Exception:
                    stage = "unknown"

                logger.warning(
                    f"[{stage}] Observation key mismatch: extra={sorted(list(extra_keys))} missing={sorted(list(missing_keys))} expected={sorted(list(self._expected_keys))} actual={sorted(list(input_keys))}"
                )
                # Also print expected keys and input keys
                logger.warning(f"Expected keys: {sorted(list(self._expected_keys))}")
                self._key_mismatch_warned = True

        # Determine which keys to actually process (intersection of available and expected)
        if self._expected_keys is not None:
            # Use expected keys that are actually available, in sorted order
            keys_to_process = sorted(self._expected_keys & input_keys)
        else:
            # Fallback: use all available keys in sorted order
            keys_to_process = sorted(input_keys)

        # Filter out keys to ignore
        keys_to_process = [k for k in keys_to_process if k not in self.keys_to_ignore]

        # Store processed keys for consistency
        self._processed_keys = keys_to_process

        feats = []
        batch_size = None

        # First pass: determine batch size from available keys
        for k in keys_to_process:
            if k in obs:
                v = obs[k]
                if isinstance(v, torch.Tensor) and v.numel() > 0:
                    batch_size = v.size(0)
                    break

        if batch_size is None:
            raise ValueError("Could not determine batch size from observations")

        # Process each observation in consistent order
        for k in keys_to_process:
            if k not in obs:
                # Create zero/dummy observation for missing keys
                print(f"Warning: Key {k} expected but not found, using zero observation")
                # This shouldn't happen since we filtered to available keys, but safety check
                continue

            v = obs[k]

            # Handle different input types and convert to consistent dtype
            if isinstance(v, torch.Tensor):
                # Ensure tensor is on the right device and dtype
                v = v.to(device=self.device, dtype=torch.float32)

                # Handle image observations
                if k in self.image_keys:
                    # All encoder types share one interface; encoders normalize/permute internally.
                    feats.append(self._encode_image(k, v))
                else:
                    # Regular observation - flatten and ensure correct dtype
                    v_flat = v.view(v.size(0), -1) if v.dim() > 1 else v.unsqueeze(0)
                    v_flat = self._encode_lowdim(k, v_flat)
                    feats.append(v_flat)

            elif isinstance(v, np.ndarray):  # TODO: this can be problematic, proceed with caution
                # Convert numpy array to tensor with consistent dtype
                v_tensor = torch.from_numpy(v).to(device=self.device, dtype=torch.float32)

                # Handle batch dimension
                if v_tensor.dim() == 1:
                    v_tensor = v_tensor.unsqueeze(0)  # Add batch dimension

                # Expand to match batch size if needed
                if v_tensor.size(0) != batch_size:
                    if v_tensor.size(0) == 1:
                        v_tensor = v_tensor.expand(batch_size, *v_tensor.shape[1:])
                    else:
                        print(f"Warning: Batch size mismatch for key {k}: {v_tensor.size(0)} vs {batch_size}")
                        continue

                # Handle image observations
                if k in self.image_keys:
                    feats.append(self._encode_image(k, v_tensor))
                else:
                    # Regular observation - flatten
                    v_flat = v_tensor.view(v_tensor.size(0), -1)
                    v_flat = self._encode_lowdim(k, v_flat)
                    feats.append(v_flat)

            elif isinstance(v, str):  # TODO: this can be problematic, proceed with caution
                # Handle language strings
                if self.use_language_embeddings and self.lang_encoder is not None:
                    try:
                        # Convert string to language embedding
                        embeddings = self.lang_encoder.encode(
                            [v] if isinstance(v, str) else v,
                            convert_to_tensor=True,
                            device=self.device,
                        )
                        # Ensure correct batch size
                        if embeddings.size(0) != batch_size:
                            embeddings = embeddings.expand(batch_size, -1)
                        embeddings = self._encode_lowdim(k, embeddings)
                        feats.append(embeddings)
                    except Exception as e:
                        print(f"Warning: Failed to encode language for {k}: {e}")
                        # Create zero embedding
                        zero_emb = torch.zeros(
                            batch_size,
                            self.lang_embedding_dim,
                            device=self.device,
                            dtype=torch.float32,
                        )
                        zero_emb = self._encode_lowdim(k, zero_emb)
                        feats.append(zero_emb)
                else:
                    print(f"Warning: Language string provided but language embeddings disabled for {k}")
                    # Create zero embedding
                    zero_emb = torch.zeros(
                        batch_size,
                        self.lang_embedding_dim,
                        device=self.device,
                        dtype=torch.float32,
                    )
                    zero_emb = self._encode_lowdim(k, zero_emb)
                    feats.append(zero_emb)

            elif isinstance(v, list):  # TODO: this can be problematic, proceed with caution
                # Handle list of tensors (e.g., from data loader)
                if all(isinstance(item, torch.Tensor) for item in v):
                    try:
                        v_stacked = torch.stack(v)
                        if v_stacked.dim() > 2:
                            v_stacked = v_stacked.to(device=self.device, dtype=torch.float32)
                        if k in self.image_keys:
                            feats.append(self._encode_image(k, v_stacked))
                        else:
                            v_flat = v_stacked.view(v_stacked.size(0), -1)
                            v_flat = self._encode_lowdim(k, v_flat)
                            feats.append(v_flat)
                    except Exception as e:
                        print(f"Warning: Failed to stack tensors for {k}: {e}")
                        continue
                else:
                    print(f"Warning: List observation {k} contains non-tensor items")
                    continue

            else:
                print(f"Warning: Unsupported observation type for {k}: {type(v)}")
                continue

        if not feats:
            raise ValueError("No valid features extracted from observations")

        # Concatenate all features - they should all be float32 now
        try:
            obs_flat = torch.cat(feats, dim=1)
        except Exception as e:
            print(f"Error concatenating features: {e}")
            print(f"Feature shapes: {[f.shape for f in feats]}")
            print(f"Feature dtypes: {[f.dtype for f in feats]}")
            raise

        # Concatenated per-modality features, projected to d_model later by obs_projection.
        return obs_flat.to(dtype=torch.float32)
