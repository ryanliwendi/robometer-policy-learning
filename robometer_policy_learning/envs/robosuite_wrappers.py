"""
Robosuite environment loader for policy learning.

Wraps native `robosuite` environments so they expose the same dict-observation
interface used by the rest of this repo (``{"state", "image", "wrist_image", ...}``),
plus optional DINOv2 image embeddings and sentence-transformer language embeddings.

The resulting vectorized environment is a drop-in for ``make_env`` in
``robometer_policy_learning.utils.env_utils`` (see the ``robosuite`` branch there).
"""

from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from transformers import AutoImageProcessor, AutoModel

from robometer_policy_learning.envs.dino_wrapper import DinoEmbeddingWrapper
from robometer_policy_learning.envs.libero_pi0_wrapper import VectorLiberoPromptWrapper
from robometer_policy_learning.utils.env_utils import GymToGymnasiumWrapper

# Best-effort natural-language instruction per robosuite task. Used to compute the
# language embedding when a sentence model is provided. Falls back to the env name.
ROBOSUITE_TASK_TO_LANG: Dict[str, str] = {
    "Lift": "lift the cube",
    "Stack": "stack the red cube on top of the green cube",
    "Door": "open the door",
    "Wipe": "wipe the table clean",
    "PickPlace": "pick up the objects and place them in the correct bins",
    "PickPlaceCan": "pick up the can and place it in the bin",
    "PickPlaceBread": "pick up the bread and place it in the bin",
    "PickPlaceMilk": "pick up the milk and place it in the bin",
    "PickPlaceCereal": "pick up the cereal box and place it in the bin",
    "NutAssembly": "fit the nuts onto the pegs",
    "NutAssemblySquare": "fit the square nut onto the square peg",
    "NutAssemblyRound": "fit the round nut onto the round peg",
    "TwoArmLift": "lift the pot using both arms",
    "TwoArmPegInHole": "insert the peg into the hole using both arms",
    "TwoArmHandover": "hand over the hammer from one arm to the other",
}


class RobosuiteObsWrapper(gym.Wrapper):
    """
    Convert native robosuite observations into the dict format used for policy learning.

    Produces observations with keys:
        - ``state``: concatenated proprioceptive (and optionally object) state, float32
        - ``<image_obs_key>``: agentview RGB image, uint8 ``(H, W, 3)``
        - ``<wrist_obs_key>``: wrist (eye-in-hand) RGB image, uint8 ``(H, W, 3)`` (if available)

    The output image key names default to ``image`` / ``wrist_image`` but can be set to
    e.g. ``agentview_image`` / ``robot0_eye_in_hand_image`` so the online observation keys
    match an offline robomimic dataset.

    robosuite renders camera images upside-down, so images are flipped vertically.
    The base env is expected to already use ``GymToGymnasiumWrapper`` so that
    ``reset``/``step`` follow the Gymnasium 5-tuple convention.
    """

    def __init__(
        self,
        env: gym.Env,
        image_size: int = 224,
        camera_name: str = "agentview",
        wrist_camera_name: Optional[str] = "robot0_eye_in_hand",
        proprio_keys: Sequence[str] = ("robot0_proprio-state",),
        use_full_state: bool = False,
        language_instruction: str = "",
        terminate_on_success: bool = False,
        flip_images: bool = True,
        image_obs_key: str = "image",
        wrist_obs_key: str = "wrist_image",
    ):
        super().__init__(env)
        self.image_size = image_size
        self.camera_name = camera_name
        self.wrist_camera_name = wrist_camera_name
        self.use_full_state = use_full_state
        self.language_instruction = language_instruction
        self.terminate_on_success = terminate_on_success
        self.flip_images = flip_images
        self.image_obs_key = image_obs_key
        self.wrist_obs_key = wrist_obs_key

        # Probe a sample observation (no env.reset needed) to discover available keys/dims.
        sample = self.env.observation_spec()

        self._image_key = f"{camera_name}_image"
        assert self._image_key in sample, (
            f"Camera '{camera_name}' not in robosuite obs (keys: {list(sample.keys())}). "
            "Pass the matching camera_name / camera_names to suite.make."
        )
        self._wrist_image_key = f"{wrist_camera_name}_image" if wrist_camera_name else None
        self._has_wrist = self._wrist_image_key is not None and self._wrist_image_key in sample

        # State is the concatenation of the requested proprio keys (+ object-state if full state).
        self._state_keys: List[str] = [k for k in proprio_keys if k in sample]
        if not self._state_keys:
            raise ValueError(
                f"None of proprio_keys={list(proprio_keys)} found in robosuite obs (keys: {list(sample.keys())})"
            )
        if self.use_full_state and "object-state" in sample and "object-state" not in self._state_keys:
            self._state_keys.append("object-state")
        state_dim = int(sum(np.asarray(sample[k]).reshape(-1).shape[0] for k in self._state_keys))

        spaces = {
            "state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32),
            self.image_obs_key: gym.spaces.Box(
                low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
            ),
        }
        if self._has_wrist:
            spaces[self.wrist_obs_key] = gym.spaces.Box(
                low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
            )
        self.observation_space = gym.spaces.Dict(spaces)

        # robosuite action space is symmetric; expose a Box matching action_spec.
        low, high = self.env.action_spec
        self.action_space = gym.spaces.Box(low=np.float32(low), high=np.float32(high), dtype=np.float32)

    def _format_image(self, img: np.ndarray) -> np.ndarray:
        if self.flip_images:
            img = img[::-1]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(img)

    def _format_obs(self, raw_obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        state = np.concatenate(
            [np.asarray(raw_obs[k], dtype=np.float32).reshape(-1) for k in self._state_keys]
        ).astype(np.float32)
        obs = {
            "state": state,
            self.image_obs_key: self._format_image(raw_obs[self._image_key]),
        }
        if self._has_wrist:
            obs[self.wrist_obs_key] = self._format_image(raw_obs[self._wrist_image_key])
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # robosuite has no seedable reset; seed numpy for reproducible task initialization.
        if seed is not None:
            np.random.seed(seed)
        raw_obs, info = self.env.reset()
        info["language_instruction"] = self.language_instruction
        return self._format_obs(raw_obs), info

    def step(self, action):
        raw_obs, reward, terminated, truncated, info = self.env.step(action)

        success = False
        try:
            success = bool(self.env._check_success())
        except AttributeError:
            success = bool(info.get("success", False))

        info["success"] = success
        info["is_success"] = success
        info["sparse_reward"] = float(success)
        info["language_instruction"] = self.language_instruction

        if self.terminate_on_success and success:
            terminated = True

        return self._format_obs(raw_obs), np.float64(reward), terminated, truncated, info

    def __getattr__(self, name):
        return getattr(self.env, name)


def setup_robosuite_env(
    env_name: str,
    n_envs: int,
    controller: str = "OSC_POSE",
    dinov2_model: Optional[AutoModel] = None,
    dinov2_processor: Optional[AutoImageProcessor] = None,
    sentence_model: Optional[SentenceTransformer] = None,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    max_episode_steps: int = 500,
    image_size: int = 224,
    image_keys: List[str] = ["image"],
    use_full_state: bool = False,
    use_dense_reward: bool = True,
    terminate_on_success: bool = False,
    chunk_size: Optional[int] = None,
    n_action_steps: int = 1,
    extra_keys_to_drop: List[str] = [],
) -> Tuple[gym.vector.VectorEnv, List[str]]:
    """
    Create a vectorized robosuite environment ready for policy learning.

    The robot is always a Franka Panda.

    Args:
        env_name: robosuite task name (e.g. "Lift", "Stack", "PickPlaceCan").
        n_envs: Number of parallel environments.
        controller: robosuite default controller (e.g. "OSC_POSE", "JOINT_VELOCITY").
        dinov2_model / dinov2_processor: optional DINOv2 model/processor for image embeddings.
        sentence_model: optional sentence transformer for language embeddings.
        device: torch device for the embedding models.
        seed: base random seed (env i is seeded with seed + i).
        max_episode_steps: episode horizon / time limit.
        image_size: rendered camera height/width (square images).
        image_keys: observation keys fed to the DINO embedding wrapper.
        use_full_state: include object-state in the proprioceptive state vector.
        use_dense_reward: use robosuite reward shaping (dense) vs. sparse rewards.
        terminate_on_success: terminate the episode as soon as the task succeeds.
        chunk_size: action chunk size (None for no chunking). When set, the vectorized env
            is wrapped with ``VectorActionChunkingWrapper`` for open-loop chunk execution.
        n_action_steps: number of actions to execute open-loop from each predicted chunk
            before replanning (must be <= chunk_size). Only used when chunk_size is set.
        extra_keys_to_drop: additional observation keys to drop in the replay buffer.

    Returns:
        env: vectorized gymnasium environment.
        remove_obs_keys: observation keys to drop before storing transitions.
    """
    try:
        import robosuite as suite
        from robosuite import load_controller_config
    except ImportError:
        logger.error("robosuite not found. Please install robosuite.")
        raise

    from robometer_policy_learning.envs.action_wrappers import VectorActionChunkingWrapper

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if seed is None:
        seed = 0

    language_instruction = ROBOSUITE_TASK_TO_LANG.get(env_name, env_name)
    logger.info(
        f"Setting up robosuite environment: {env_name} (robots=Panda, controller={controller}, "
        f"reward={'dense' if use_dense_reward else 'sparse'})"
    )

    controller_config = load_controller_config(default_controller=controller)
    camera_name = "agentview"
    wrist_camera_name = "robot0_eye_in_hand"

    env_fns = []
    for i in range(n_envs):

        def make_env(rank=i):
            base_env = suite.make(
                env_name=env_name,
                robots="Panda",
                controller_configs=controller_config,
                has_renderer=False,
                has_offscreen_renderer=True,
                use_camera_obs=True,
                use_object_obs=True,
                camera_names=[camera_name, wrist_camera_name],
                camera_heights=image_size,
                camera_widths=image_size,
                reward_shaping=use_dense_reward,
                horizon=max_episode_steps,
                ignore_done=False,
                hard_reset=False,
            )
            # Seed numpy for reproducible per-env task initialization.
            np.random.seed(seed + rank)

            env = GymToGymnasiumWrapper(base_env, time_limit=max_episode_steps)
            env = RobosuiteObsWrapper(
                env,
                image_size=image_size,
                camera_name=camera_name,
                wrist_camera_name=wrist_camera_name,
                use_full_state=use_full_state,
                language_instruction=language_instruction,
                terminate_on_success=terminate_on_success,
            )

            if dinov2_model is not None:
                env = DinoEmbeddingWrapper(
                    env, dinov2_model, dinov2_processor, device=device, image_keys=image_keys
                )

            # Metadata used by downstream prompt wrapper / workers.
            env.language_instruction = language_instruction
            env.task_id = env_name
            return env

        env_fns.append(make_env)

    env = gym.vector.SyncVectorEnv(env_fns)
    # Adds 'prompt' (and 'language' embedding when sentence_model is provided) to observations.
    env = VectorLiberoPromptWrapper(env, sentence_model)

    # Open-loop action chunking on the vectorized env (exposes is_chunk_empty /
    # _get_last_action so the rollout/eval workers execute n_action_steps per chunk).
    if chunk_size is not None:
        env = VectorActionChunkingWrapper(env, chunk_size=chunk_size, n_action_steps=n_action_steps)

    logger.info(f"✓ Created {n_envs} robosuite '{env_name}' environment(s)")

    remove_obs_keys = ["wrist_image", "language", "prompt"] + extra_keys_to_drop
    if dinov2_model is not None:
        remove_obs_keys += image_keys
    return env, remove_obs_keys


def load_robomimic_env_metadata(dataset_path: str) -> Dict:
    """
    Fetch the robosuite environment metadata stored inside a robomimic ``.h5`` dataset.

    robomimic datasets serialize the exact environment used to collect the demonstrations
    under ``data.attrs["env_args"]``. Reconstructing the environment from this metadata
    guarantees that the online RL / evaluation environment matches the one the offline
    data was collected in (same task, robot, controller, control frequency, etc.).

    Args:
        dataset_path: path to the robomimic ``.h5`` demonstration file.

    Returns:
        env_meta: dict with keys ``"env_name"``, ``"type"`` and ``"env_kwargs"``.
    """
    try:
        import robomimic.utils.file_utils as FileUtils
    except ImportError:
        logger.error(
            "robomimic not found. Install it into the uv environment, e.g. "
            "`uv add robomimic` (or `uv pip install robomimic`)."
        )
        raise

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    logger.info(
        f"Loaded robomimic env metadata from {dataset_path}: "
        f"env_name={env_meta.get('env_name')}, type={env_meta.get('type')}"
    )
    return env_meta


def setup_robomimic_env(
    dataset_path: str,
    n_envs: int,
    dinov2_model: Optional[AutoModel] = None,
    dinov2_processor: Optional[AutoImageProcessor] = None,
    sentence_model: Optional[SentenceTransformer] = None,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    max_episode_steps: int = 500,
    image_size: Optional[int] = None,
    image_keys: List[str] = ["agentview_image"],
    use_full_state: bool = False,
    use_dense_reward: bool = True,
    terminate_on_success: bool = False,
    chunk_size: Optional[int] = None,
    n_action_steps: int = 1,
    extra_keys_to_drop: List[str] = [],
    camera_name: str = "agentview",
    wrist_camera_name: Optional[str] = "robot0_eye_in_hand",
    language_instruction: Optional[str] = None,
) -> Tuple[gym.vector.VectorEnv, List[str]]:
    """
    Create a vectorized robosuite environment reconstructed from a robomimic dataset.

    Unlike :func:`setup_robosuite_env`, the task / robot / controller configuration is
    read from the robomimic ``.h5`` metadata (so the environment matches the offline
    data exactly). Only the rendering / camera / horizon settings are overridden so the
    env produces the dict observations (``state`` / ``image`` / ``wrist_image``) used by
    the rest of this repo.

    Args:
        dataset_path: path to the robomimic ``.h5`` demonstration file.
        n_envs: number of parallel environments.
        dinov2_model / dinov2_processor: optional DINOv2 model/processor for image embeddings.
        sentence_model: optional sentence transformer for language embeddings.
        device: torch device for the embedding models.
        seed: base random seed (env i is seeded with seed + i).
        max_episode_steps: episode horizon / time limit.
        image_size: rendered camera height/width (square images). If None, defaults to the
            dataset's camera resolution (so rendered images match the offline data, which
            keeps image-derived features such as DINO embeddings aligned).
        image_keys: observation keys fed to the DINO embedding wrapper.
        use_full_state: include object-state in the proprioceptive state vector.
        use_dense_reward: use robosuite reward shaping (dense) vs. sparse rewards.
        terminate_on_success: terminate the episode as soon as the task succeeds.
        chunk_size: action chunk size (None for no chunking). When set, the vectorized env
            is wrapped with ``VectorActionChunkingWrapper`` for open-loop chunk execution.
        n_action_steps: number of actions to execute open-loop from each predicted chunk
            before replanning (must be <= chunk_size). Only used when chunk_size is set.
        extra_keys_to_drop: additional observation keys to drop in the replay buffer.
        camera_name: third-person camera used for the ``image`` observation.
        wrist_camera_name: eye-in-hand camera used for ``wrist_image`` (None to disable).
        language_instruction: override the natural-language instruction. Defaults to the
            best-effort mapping in ``ROBOSUITE_TASK_TO_LANG`` keyed by the dataset env name.

    Returns:
        env: vectorized gymnasium environment.
        remove_obs_keys: observation keys to drop before storing transitions.
    """
    try:
        import robosuite as suite
    except ImportError:
        logger.error("robosuite not found. Please install robosuite.")
        raise

    from robometer_policy_learning.envs.action_wrappers import VectorActionChunkingWrapper

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if seed is None:
        seed = np.random.randint(1, 100)

    env_meta = load_robomimic_env_metadata(dataset_path)
    env_name = env_meta["env_name"]
    if language_instruction is None:
        language_instruction = ROBOSUITE_TASK_TO_LANG.get(env_name, env_name)

    # Start from the dataset's env kwargs (robots, controller_configs, control_freq, ...)
    # and override only what we need for image observations / horizon / rewards.
    env_kwargs = deepcopy(env_meta.get("env_kwargs", {}))
    # `env_name` is passed positionally to suite.make; drop any duplicate in kwargs.
    env_kwargs.pop("env_name", None)

    # Default the render resolution to the dataset's camera size so rendered images match
    # the offline frames (keeps image-derived features like DINO embeddings aligned).
    if image_size is None:
        dataset_image_size = env_kwargs.get("camera_heights", env_kwargs.get("camera_widths"))
        image_size = int(dataset_image_size) if dataset_image_size is not None else 224
        logger.info(f"image_size not specified; using dataset camera size {image_size}.")

    camera_names = [camera_name]
    if wrist_camera_name and wrist_camera_name != camera_name:
        camera_names.append(wrist_camera_name)

    env_kwargs.update(
        dict(
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            use_object_obs=True,
            camera_names=camera_names,
            camera_heights=image_size,
            camera_widths=image_size,
            camera_depths=False,
            reward_shaping=use_dense_reward,
            horizon=max_episode_steps,
            ignore_done=False,
            hard_reset=False,
        )
    )

    robots = env_kwargs.get("robots", "Panda")
    # Output image obs keys match the robomimic dataset's camera obs key naming.
    image_obs_key = f"{camera_name}_image"
    wrist_obs_key = f"{wrist_camera_name}_image" if wrist_camera_name else "wrist_image"
    logger.info(
        f"Setting up robomimic environment: {env_name} (robots={robots}, "
        f"reward={'dense' if use_dense_reward else 'sparse'}) from {dataset_path}"
    )

    env_fns = []
    for i in range(n_envs):

        def make_env(rank=i):
            base_env = suite.make(env_name, **deepcopy(env_kwargs))
            # Seed numpy for reproducible per-env task initialization.
            np.random.seed(seed + rank)

            env = GymToGymnasiumWrapper(base_env, time_limit=max_episode_steps)
            env = RobosuiteObsWrapper(
                env,
                image_size=image_size,
                camera_name=camera_name,
                wrist_camera_name=wrist_camera_name,
                use_full_state=use_full_state,
                language_instruction=language_instruction,
                terminate_on_success=terminate_on_success,
                # Name the image obs keys to match the robomimic dataset's camera obs keys
                # (e.g. "agentview_image" / "robot0_eye_in_hand_image") so the online env
                # observations align with the converted offline dataset.
                image_obs_key=image_obs_key,
                wrist_obs_key=wrist_obs_key,
            )

            if dinov2_model is not None:
                env = DinoEmbeddingWrapper(
                    env, dinov2_model, dinov2_processor, device=device, image_keys=image_keys
                )

            # Metadata used by downstream prompt wrapper / workers.
            env.language_instruction = language_instruction
            env.task_id = env_name
            return env

        env_fns.append(make_env)

    env = gym.vector.SyncVectorEnv(env_fns)
    # Adds 'prompt' (and 'language' embedding when sentence_model is provided) to observations.
    env = VectorLiberoPromptWrapper(env, sentence_model)

    # Open-loop action chunking on the vectorized env (exposes is_chunk_empty /
    # _get_last_action so the rollout/eval workers execute n_action_steps per chunk).
    if chunk_size is not None:
        env = VectorActionChunkingWrapper(env, chunk_size=chunk_size, n_action_steps=n_action_steps)

    logger.info(
        f"✓ Created {n_envs} robomimic '{env_name}' environment(s) "
        f"(image_size={image_size}, image_obs_key='{image_obs_key}', "
        f"state='robot0_proprio-state'{' + object-state' if use_full_state else ''})"
    )

    remove_obs_keys = [wrist_obs_key, "language", "prompt"] + extra_keys_to_drop
    if dinov2_model is not None:
        remove_obs_keys += image_keys
    return env, remove_obs_keys
