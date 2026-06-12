"""
Transition-level transforms for modifying individual transitions.

These transforms operate on single Transition objects and can be applied
during sampling to modify rewards, observations, or other transition data.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, Sequence, Tuple
from robometer_policy_learning.buffers.base_replay_buffer import Transition
from robometer_policy_learning.modules.transformer.transformer_utils import identify_image_keys


class TransitionTransform:
    """
    Base class for transforms that operate on individual transitions.
    This is more flexible than batch-level transforms.
    """

    def __call__(self, transition: Transition) -> Transition:
        """Apply transform to a single transition."""
        raise NotImplementedError

    # By default, transforms do not support batched input
    supports_batch = False


class MonotonicRewardTransform(TransitionTransform):
    """
    Transform that relabels rewards to be monotonically increasing from 0 to 1
    based on progress through the episode.

    This is useful for sparse reward environments where you want to provide
    dense reward signal based on temporal progress.
    """

    def __init__(
        self,
        mode: str = "linear",  # "linear", "quadratic", "exponential"
        success_bonus: float = 0.0,  # Additional bonus for successful episodes
        use_success_only: bool = False,
    ):  # Only apply to successful episodes
        """
        Args:
            mode: Type of monotonic increase ("linear", "quadratic", "exponential")
            success_bonus: Additional reward bonus for the final step of successful episodes
            use_success_only: If True, only apply monotonic rewards to successful episodes
        """
        self.mode = mode
        self.success_bonus = success_bonus
        self.use_success_only = use_success_only

    def __call__(self, transition: Transition) -> Transition:
        """
        Apply monotonic reward relabeling to a single transition.

        Args:
            transition: Input transition

        Returns:
            Modified transition with relabeled reward
        """
        if transition.max_steps_in_episode is None or transition.step_in_episode is None:
            # Can't apply monotonic reward without episode length info
            return transition

        # Check if this is a successful episode (if filtering is enabled)
        if self.use_success_only:
            # Assume success if the episode ends with done=True and reward > 0
            # You might want to customize this logic based on your environment
            is_successful = transition.done and transition.reward > 0
            if not is_successful:
                return transition

        # Calculate progress through episode (0 to 1)
        progress = transition.step_in_episode / (transition.max_steps_in_episode - 1)
        # if transition.done:
        #     print(transition.step_in_episode, transition.max_steps_in_episode)
        #     breakpoint()
        # if transition.max_steps_in_episode == (transition.step_in_episode - 1):
        #     breakpoint()
        progress = np.clip(progress, 0.0, 1.0)

        # Apply monotonic transformation
        if self.mode == "linear":
            new_reward = progress
        elif self.mode == "quadratic":
            new_reward = progress**2
        elif self.mode == "exponential":
            # Exponential growth: e^(progress) - 1, normalized to [0, 1]
            new_reward = (np.exp(progress) - 1) / (np.e - 1)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Add success bonus for final step if applicable
        if transition.done and self.success_bonus > 0:
            new_reward += self.success_bonus

        # Create new transition with modified reward using replace helper
        return transition.replace(reward=new_reward)

    # This transform can be applied in batch by falling back to per-transition
    supports_batch = False


class SuccessBonusTransform(TransitionTransform):
    """
    Transform that adds a bonus reward to successful episodes.

    This is equivalent to the success_bonus function but implemented
    as a transition-level transform for consistency.
    """

    def __init__(self, bonus_value: float = 10.0, debug: bool = False):
        """
        Args:
            bonus_value: Bonus reward to add to successful episodes
            debug: Whether to print debug information about reward transformations
        """
        self.bonus_value = bonus_value
        self.debug = debug

    def __call__(self, transition: Transition) -> Transition:
        """
        Add success bonus to successful episodes.

        Args:
            transition: Input transition

        Returns:
            Modified transition with success bonus
        """
        # Add bonus to successful episodes (done=True)
        if transition.done:
            new_reward = transition.reward + self.bonus_value
            if self.debug:
                print(
                    f"SuccessBonusTransform: original_reward={transition.reward}, bonus={self.bonus_value}, new_reward={new_reward}"
                )
        else:
            new_reward = transition.reward

        # Create new transition with modified reward using replace helper
        return transition.replace(reward=new_reward)

    # This transform can be applied in batch by falling back to per-transition
    supports_batch = False

    def __repr__(self):
        return f"SuccessBonusTransform(bonus_value={self.bonus_value}, debug={self.debug})"


# Legacy batch-level transform functions for backward compatibility
def success_bonus(bonus_value):
    """
    Legacy success bonus transform that now works with Transition objects.

    This is kept for backward compatibility with existing code.
    For new code, consider using SuccessBonusTransform directly.
    """

    def _success_bonus_transform(transition: Transition) -> Transition:
        # Add bonus to successful episodes (done=True)
        if transition.done:
            new_reward = transition.reward + bonus_value
        else:
            new_reward = transition.reward

        # Create new transition with modified reward using replace helper
        return transition.replace(reward=new_reward)

    return _success_bonus_transform

class ImageAugmentationTransform(TransitionTransform):
    """Image-only random crop and photometric jitter for transition post_transforms."""
    supports_batch = False

    def __init__(self, observation_space: Optional[Any] = None,
                 random_crop: bool = True, 
                 crop_prob: float = 0.3, 
                 crop_scale: Tuple[float, float] = (0.97, 1.0), 
                 photometric_prob: float = 0.5, 
                 brightness: float = 0.05, 
                 contrast: float = 0.05, 
                 saturation: float = 0.03, 
                 gamma: float = 0.03, 
                 apply_to_next_obs: bool = True, 
                 same_transform_on_next_obs: bool = True, 
                 min_spatial_size: int = 16, 
                 seed: Optional[int] = None):

        if observation_space is not None:
            obs_keys = list(observation_space.spaces.keys()) if hasattr(observation_space, "spaces") else []
            self.image_keys = set(identify_image_keys(obs_keys))
        self.random_crop = random_crop
        self.crop_prob = crop_prob
        self.crop_scale = crop_scale
        self.photometric_prob = photometric_prob
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.gamma = gamma
        self.apply_to_next_obs = apply_to_next_obs
        self.same_transform_on_next_obs = same_transform_on_next_obs
        self.min_spatial_size = min_spatial_size
        self.rng = np.random.default_rng(seed)
        if crop_scale[0] <= 0 or crop_scale[1] <= 0 or crop_scale[0] > crop_scale[1]:
            raise ValueError(f"Invalid crop_scale={crop_scale}; expected positive (min, max)")

    def __call__(self, transition: Transition) -> Transition:
        obs = dict(transition.obs)
        next_obs = dict(transition.next_obs)
        keys = set(obs.keys())
        if self.apply_to_next_obs:
            keys.update(next_obs.keys())
        for key in keys:
            if key in obs and self._should_augment_key(key, obs[key]):
                params = self._sample_params(obs[key])
                original_obs_shape = tuple(obs[key].shape)
                obs[key] = self._augment_image(obs[key], params)
                if self.apply_to_next_obs and key in next_obs and self._should_augment_key(key, next_obs[key]):
                    next_params = params
                    if not self.same_transform_on_next_obs or original_obs_shape != tuple(next_obs[key].shape):
                        next_params = self._sample_params(next_obs[key])
                    next_obs[key] = self._augment_image(next_obs[key], next_params)
            elif self.apply_to_next_obs and key in next_obs and self._should_augment_key(key, next_obs[key]):
                next_obs[key] = self._augment_image(next_obs[key], self._sample_params(next_obs[key]))
        return transition.replace(obs=obs, next_obs=next_obs)

    def _should_augment_key(self, key: str, value: Any) -> bool:
        if self.image_keys is not None and key not in self.image_keys:
            return False
        return self._infer_layout(value) is not None

    def _infer_layout(self, value: Any) -> Optional[str]:
        if not isinstance(value, (np.ndarray, torch.Tensor)):
            return None
        shape = tuple(value.shape)

        if len(shape) == 3:
            if shape[-1] in (1, 3, 4) and min(shape[0], shape[1]) >= self.min_spatial_size:
                return "HWC"
            if shape[0] in (1, 3, 4) and min(shape[1], shape[2]) >= self.min_spatial_size:
                return "CHW"

        if len(shape) == 4:
            if shape[-1] in (1, 3, 4) and min(shape[1], shape[2]) >= self.min_spatial_size:
                return "NHWC"
            if shape[1] in (1, 3, 4) and min(shape[2], shape[3]) >= self.min_spatial_size:
                return "NCHW"

        return None

    def _sample_params(self, value: Any) -> Dict[str, Any]:
        layout = self._infer_layout(value)
        if layout in ("HWC", "NHWC"):
            h, w = value.shape[-3], value.shape[-2]
        elif layout in ("CHW", "NCHW"):
            h, w = value.shape[-2], value.shape[-1]
        else:
            raise ValueError("Cannot sample image augmentation params for non-image value")
        crop = None
        if self.random_crop and self.rng.random() < self.crop_prob:
            scale = float(self.rng.uniform(self.crop_scale[0], self.crop_scale[1]))
            crop_h = max(1, min(h, int(round(h * scale))))
            crop_w = max(1, min(w, int(round(w * scale))))
            top = int(self.rng.integers(0, h - crop_h + 1)) if h > crop_h else 0
            left = int(self.rng.integers(0, w - crop_w + 1)) if w > crop_w else 0
            crop = (top, left, crop_h, crop_w)
        photometric = None
        if self.rng.random() < self.photometric_prob:
            photometric = {"brightness": self._sample_factor(self.brightness), "contrast": self._sample_factor(self.contrast), "saturation": self._sample_factor(self.saturation), "gamma": max(1e-3, self._sample_factor(self.gamma))}

        return {"layout": layout, "crop": crop, "photometric": photometric}

    def _sample_factor(self, strength: float) -> float:
        if strength <= 0:
            return 1.0

        return float(self.rng.uniform(max(0.0, 1.0 - strength), 1.0 + strength))

    def _augment_image(self, value: Any, params: Dict[str, Any]) -> Any:
        tensor, meta = self._to_nchw_float(value, params["layout"])
        if params["crop"] is not None:
            top, left, crop_h, crop_w = params["crop"]
            h, w = tensor.shape[-2], tensor.shape[-1]
            top = min(top, max(0, h - 1))
            left = min(left, max(0, w - 1))
            crop_h = min(crop_h, h - top)
            crop_w = min(crop_w, w - left)
            tensor = tensor[..., top : top + crop_h, left : left + crop_w]
            tensor = F.interpolate(tensor, size=meta["spatial_shape"], mode="bilinear", align_corners=False)
        if params["photometric"] is not None:
            tensor = self._apply_photometric(tensor, params["photometric"])

        return self._from_nchw_float(tensor, meta)

    def _to_nchw_float(self, value: Any, layout: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        is_torch = isinstance(value, torch.Tensor)
        original_dtype = value.dtype
        original_device = value.device if is_torch else None
        tensor = value.detach().clone() if is_torch else torch.from_numpy(np.asarray(value)).clone()
        if layout == "HWC":
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        elif layout == "CHW":
            tensor = tensor.unsqueeze(0)
        elif layout == "NHWC":
            tensor = tensor.permute(0, 3, 1, 2)
        elif layout != "NCHW":
            raise ValueError(f"Unsupported image layout: {layout}")
        tensor = tensor.to(dtype=torch.float32)
        value_scale = self._infer_value_scale(value)
        tensor = (tensor / value_scale).clamp(0.0, 1.0)

        return tensor, {"is_torch": is_torch, "original_dtype": original_dtype, "original_device": original_device, "layout": layout, "value_scale": value_scale, "spatial_shape": tuple(tensor.shape[-2:])}

    def _from_nchw_float(self, tensor: torch.Tensor, meta: Dict[str, Any]) -> Any:
        tensor = tensor.clamp(0.0, 1.0) * meta["value_scale"]

        if meta["layout"] == "HWC":
            tensor = tensor.squeeze(0).permute(1, 2, 0)
        elif meta["layout"] == "CHW":
            tensor = tensor.squeeze(0)
        elif meta["layout"] == "NHWC":
            tensor = tensor.permute(0, 2, 3, 1)

        if meta["is_torch"]:
            if not torch.is_floating_point(torch.empty((), dtype=meta["original_dtype"])):
                tensor = tensor.round()
            tensor = tensor.to(dtype=meta["original_dtype"])

            if meta["original_device"] is not None:
                tensor = tensor.to(meta["original_device"])
            return tensor

        np_value = tensor.cpu().numpy()
        original_dtype = meta["original_dtype"]
        if np.issubdtype(original_dtype, np.integer):
            info = np.iinfo(original_dtype)
            np_value = np.clip(np.rint(np_value), info.min, info.max)

        return np_value.astype(original_dtype, copy=False)

    def _apply_photometric(self, tensor: torch.Tensor, params: Dict[str, float]) -> torch.Tensor:
        if params["brightness"] != 1.0:
            tensor = tensor * params["brightness"]

        if params["contrast"] != 1.0:
            mean = tensor.mean(dim=(-3, -2, -1), keepdim=True)
            tensor = (tensor - mean) * params["contrast"] + mean

        if params["saturation"] != 1.0 and tensor.shape[1] >= 3:
            rgb = tensor[:, :3]
            weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
            gray = (rgb * weights).sum(dim=1, keepdim=True)
            tensor = tensor.clone()
            tensor[:, :3] = gray + (rgb - gray) * params["saturation"]

        if params["gamma"] != 1.0:
            tensor = tensor.clamp(0.0, 1.0).pow(params["gamma"])

        return tensor.clamp(0.0, 1.0)

    def _infer_value_scale(self, value: Any) -> float:
        if isinstance(value, torch.Tensor):
            if not torch.is_floating_point(value):
                return 255.0
            max_value = float(value.detach().max().cpu()) if value.numel() else 1.0
        else:
            arr = np.asarray(value)
            if np.issubdtype(arr.dtype, np.integer):
                return 255.0
            max_value = float(np.nanmax(arr)) if arr.size else 1.0

        return 255.0 if max_value > 1.5 else 1.0
