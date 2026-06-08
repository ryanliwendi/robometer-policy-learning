import numpy as np
from typing import List, Dict, Optional, Any, Tuple

from loguru import logger
from collections import deque
from robometer_policy_learning.buffers.replay_buffer import ReplayBuffer
from robometer_policy_learning.buffers.h5_replay_buffer import H5ReplayBuffer

from robometer_policy_learning.utils.robometer_utils import (
    extract_rewards_from_output,
    extract_success_probs_from_output,
    extract_rewards_from_server_output,
)
from robometer_policy_learning.utils.gpu_utils import convert_to_numpy
from robometer.evals.eval_utils import raw_dict_to_sample, build_payload, post_batch_npy
from robometer.evals.eval_server import process_batch_helper
from transformers import AutoModel, AutoImageProcessor
from sentence_transformers import SentenceTransformer
from robometer.utils.setup_utils import setup_batch_collator
from tqdm import tqdm


class RobometerReplayBuffer(ReplayBuffer):
    """
    Robometer replay buffer for storing and sampling experience transitions.
    Rewards are estimated before adding the transition to the buffer.
    """

    def __init__(
        self,
        reward_model=None,
        reward_model_config=None,
        use_relative_rewards: bool = False,
        use_eval_server: bool = False,
        eval_server_url: Optional[str] = None,
        eval_server_timeout: float = 120.0,
        reward_relabeling_keys: List[str] = ["image"],
        use_success_detection: bool = False,
        success_detection_duration: int = 2,
        success_detection_threshold: float = 0.65,
        add_estimated_reward: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reward_model = reward_model
        self.use_eval_server = use_eval_server
        self.eval_server_url = eval_server_url
        self.eval_server_timeout = eval_server_timeout
        self.reward_relabeling_keys = reward_relabeling_keys
        self.use_success_detection = use_success_detection
        self.success_detection_duration = success_detection_duration
        self.success_detection_threshold = success_detection_threshold
        self.add_estimated_reward = add_estimated_reward

        # Set max_frames once from config
        if reward_model_config is not None:
            self.max_frames = getattr(reward_model_config.data, "max_frames", 16)
        else:
            self.max_frames = 16

        if self.reward_model is not None:
            self.reward_model_config = reward_model_config
            self.processor = getattr(reward_model, "processor", None)
            self.tokenizer = getattr(reward_model, "tokenizer", None)
            if self.processor is None or self.tokenizer is None:
                raise ValueError(
                    "processor and tokenizer must be available on reward_model (reward_model.processor / reward_model.tokenizer)"
                )
            # Ensure use_multi_image is True for reward relabeling (process frames as images, not video)
            if not self.reward_model_config.data.use_multi_image:
                print("Warning: use_multi_image is False in config. Setting to True for reward relabeling.")
                self.reward_model_config.data.use_multi_image = True

            # Set up batch collator with inference=True for evaluation
            self.batch_collator = setup_batch_collator(
                self.processor, self.tokenizer, self.reward_model_config, is_eval=True
            )
        elif self.use_eval_server:
            if self.eval_server_url is None:
                raise ValueError("eval_server_url must be provided when use_eval_server=True")
            logger.info(f"Using eval_server at {self.eval_server_url} for reward computation")

        self.use_relative_rewards = use_relative_rewards
        if self.use_relative_rewards:
            self.prev_reward = {key: 0.0 for key in self.reward_relabeling_keys}
        self.success_tracker = {
            key: deque(maxlen=self.success_detection_duration) for key in self.reward_relabeling_keys
        }

    def _compute_reward_single(self, raw_data: Dict[str, Any]) -> Tuple[float, float]:
        """
        Compute reward for a single sample using either local reward model or eval_server.

        Args:
            raw_data: Dictionary containing frames, task, video_embeddings, text_embedding, etc.

        Returns:
            Reward value as float
        """
        if self.reward_model is not None:
            # Use local reward model
            sample = raw_dict_to_sample(
                raw_data=raw_data,
                max_frames=self.max_frames,
                sample_type="progress",
            )

            is_discrete_mode = self.reward_model_config.loss.progress_loss_type == "discrete"
            progress_discrete_bins = self.reward_model_config.loss.progress_discrete_bins
            outputs = process_batch_helper(
                model_type=self.reward_model_config.model.model_type,
                model=self.reward_model,
                tokenizer=self.tokenizer,
                batch_collator=self.batch_collator,
                device=self.reward_model.device,
                batch_data=[sample.model_dump()],
                job_id=0,
                is_discrete_mode=is_discrete_mode,
                num_bins=progress_discrete_bins,
            )
        elif self.use_eval_server:
            # Use eval_server
            sample = raw_dict_to_sample(
                raw_data=raw_data,
                max_frames=self.max_frames,
                sample_type="progress",
            )

            files, sample_data = build_payload([sample])
            outputs = post_batch_npy(self.eval_server_url, files, sample_data, timeout_s=self.eval_server_timeout)
        else:
            raise ValueError("Neither reward_model nor use_eval_server is set")

        rewards = extract_rewards_from_output(outputs)
        suceess_probs = extract_success_probs_from_output(outputs)
        return float(rewards[0]), float(suceess_probs[0])

    def _compute_rewards_batch(self, batch_raw: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
        """
        Compute rewards for a batch of samples using either local reward model or eval_server.

        Args:
            batch_raw: List of dictionaries, each containing frames, task, video_embeddings, text_embedding, etc.

        Returns:
            Tuple of (List of reward values as floats, List of success probabilities as floats)
        """
        if self.reward_model is not None:
            # Use local reward model
            samples = [
                raw_dict_to_sample(
                    raw_data=raw_data_item,
                    max_frames=self.max_frames,
                    sample_type="progress",
                )
                for raw_data_item in batch_raw
            ]

            is_discrete_mode = self.reward_model_config.loss.progress_loss_type == "discrete"
            progress_discrete_bins = self.reward_model_config.loss.progress_discrete_bins
            outputs = process_batch_helper(
                model_type=self.reward_model_config.model.model_type,
                model=self.reward_model,
                tokenizer=self.tokenizer,
                batch_collator=self.batch_collator,
                device=self.reward_model.device,
                batch_data=[sample.model_dump() for sample in samples],
                job_id=0,
                is_discrete_mode=is_discrete_mode,
                num_bins=progress_discrete_bins,
            )
        elif self.use_eval_server:
            # Use eval_server
            samples = [
                raw_dict_to_sample(
                    raw_data=raw_data_item,
                    max_frames=self.max_frames,
                    sample_type="progress",
                )
                for raw_data_item in batch_raw
            ]

            files, sample_data = build_payload(samples)
            outputs = post_batch_npy(self.eval_server_url, files, sample_data, timeout_s=self.eval_server_timeout)
        else:
            raise ValueError("Neither reward_model nor use_eval_server is set")

        rewards_batch = extract_rewards_from_output(outputs)
        success_probs_batch = extract_success_probs_from_output(outputs)
        return rewards_batch.tolist(), success_probs_batch.tolist()

    def _add(
        self,
        language_instruction=None,
        video_frames=None,
        dino_embeddings=None,
        text_embedding=None,
        **kwargs,
    ):
        # Calculate reward using reward model or eval_server if available
        if self.reward_model is not None or self.use_eval_server:
            # Ensure text_embedding is a numpy array
            text_emb = convert_to_numpy(text_embedding)
            # Ensure dino_embeddings is a numpy array
            dino_embeddings = convert_to_numpy(dino_embeddings)
            avg_reward = 0.0
            for index, key in enumerate(self.reward_relabeling_keys):
                # Convert embeddings to proper format (common for both paths)
                if isinstance(dino_embeddings, list) and len(dino_embeddings) > 0:
                    # Convert list of embeddings to array [T, D]
                    dino_embeddings = np.array(dino_embeddings)
                # Take subset for this key: each key's embedding is a chunk of the list
                # since dino embeddings for each key are concatenated.
                video_embeddings_array = None
                if len(dino_embeddings) > 0:
                    embeddings_per_key = dino_embeddings.shape[1] // len(self.reward_relabeling_keys)
                    video_embeddings_array = dino_embeddings[
                        :, index * embeddings_per_key : (index + 1) * embeddings_per_key
                    ]

                raw_data = dict(
                    frames=np.array(video_frames[key]) if video_frames[key] is not None else np.array([]),
                    task=language_instruction,
                    id=kwargs.get("episode_id"),
                    metadata=dict(
                        subsequence_length=len(video_frames[key]) if video_frames[key] is not None else 0,
                    ),
                    video_embeddings=video_embeddings_array,
                    text_embedding=text_emb,
                )

                reward, success_prob = self._compute_reward_single(raw_data)
                self.success_tracker[key].append(success_prob)

                # Apply relative rewards if enabled
                if self.use_relative_rewards:
                    current_reward = reward
                    reward = reward - self.prev_reward[key]
                    # Store original absolute reward
                    self.prev_reward[key] = current_reward
                    if kwargs.get("done") or kwargs.get("truncated"):
                        self.prev_reward[key] = 0.0
                avg_reward += reward

            avg_reward /= len(self.reward_relabeling_keys)
            if self.add_estimated_reward:
                kwargs["reward"] += avg_reward
            else:
                kwargs["reward"] = avg_reward
            if self.use_success_detection:
                # Check if the episode is done based on majority vote of success probabilities
                vote = 0
                for key in self.reward_relabeling_keys:
                    for success_prob in self.success_tracker[key]:
                        if success_prob > float(self.success_detection_threshold):
                            vote += 1
                if vote > (len(self.reward_relabeling_keys) * self.success_detection_duration / 2):
                    kwargs["done"] = True
                if kwargs["done"] or kwargs["truncated"]:
                    for key in self.reward_relabeling_keys:
                        self.success_tracker[key].clear()

        super()._add(**kwargs)


class RobometerH5ReplayBuffer(H5ReplayBuffer):
    """Offline HDF5 buffer that relabels rewards with a Robometer reward model / eval server.

    DINO image embeddings and sentence (language) embeddings are handled by the base
    :class:`H5ReplayBuffer`; this subclass only adds reward relabeling on top. Use it only
    when reward relabeling is required (``reward_model`` or ``use_eval_server``); for plain
    embedding-augmented offline data, use :class:`H5ReplayBuffer` directly.
    """

    def __init__(
        self,
        reward_model=None,
        reward_model_config=None,
        use_relative_rewards: bool = False,
        use_eval_server: bool = False,
        eval_server_url: Optional[str] = None,
        eval_server_timeout: float = 120.0,
        sentence_model: SentenceTransformer = None,
        dinov2_model: AutoModel = None,
        dinov2_processor: AutoImageProcessor = None,
        reward_relabeling_keys: List[str] = ["image"],
        use_success_detection: bool = False,
        success_detection_duration: int = 2,
        success_detection_threshold: float = 0.65,
        add_estimated_reward: bool = False,
        **kwargs,
    ):
        # Reward relabeling embeds the same image keys it scores, so the base buffer
        # computes DINO embeddings over `reward_relabeling_keys` (set before super()).
        self.reward_relabeling_keys = reward_relabeling_keys
        super().__init__(
            dinov2_model=dinov2_model,
            dinov2_processor=dinov2_processor,
            sentence_model=sentence_model,
            dino_embedding_keys=reward_relabeling_keys,
            **kwargs,
        )
        # Base has now computed DINO/language embeddings and added them to obs.

        self.use_eval_server = use_eval_server
        self.eval_server_url = eval_server_url
        self.eval_server_timeout = eval_server_timeout
        self.use_success_detection = use_success_detection
        self.success_detection_duration = success_detection_duration
        self.success_detection_threshold = success_detection_threshold
        self.add_estimated_reward = add_estimated_reward
        self.use_relative_rewards = use_relative_rewards

        # Set max_frames once from config
        if reward_model_config is not None:
            self.max_frames = getattr(reward_model_config.data, "max_frames", 16)
        else:
            self.max_frames = 16

        self.reward_model = reward_model
        if self.reward_model is not None:
            self.reward_model_config = reward_model_config
            self.processor = getattr(reward_model, "processor", None)
            self.tokenizer = getattr(reward_model, "tokenizer", None)
            if self.processor is None or self.tokenizer is None:
                raise ValueError(
                    "processor and tokenizer must be available on reward_model (reward_model.processor / reward_model.tokenizer)"
                )
            # Ensure use_multi_image is True for reward relabeling (process frames as images, not video)
            if not self.reward_model_config.data.use_multi_image:
                logger.warning("use_multi_image is False in config. Setting to True for reward relabeling.")
                self.reward_model_config.data.use_multi_image = True

            # Set up batch collator with inference=True for evaluation
            self.batch_collator = setup_batch_collator(
                self.processor, self.tokenizer, self.reward_model_config, is_eval=True
            )
        elif self.use_eval_server:
            if self.eval_server_url is None:
                raise ValueError("eval_server_url must be provided when use_eval_server=True")
            logger.info(f"Using eval_server at {self.eval_server_url} for reward computation")

        # Relabel rewards using the reward model (embeddings already attached by the base).
        if self.reward_model is not None:
            self.relabel_rewards(verbose=True, batch_size=8)

    def relabel_rewards(self, verbose: bool = True, batch_size: Optional[int] = None):
        """
        Relabel rewards in the HDF5 cache using the reward model.

        This function iterates through all demos in the HDF5 cache, extracts video frames
        and language instructions, processes them through the reward model, and updates
        the cached rewards with the model's predictions.

        Args:
            verbose: Whether to print progress information
            batch_size: Batch size for reward model inference
        """
        logger.info("Starting to relabel rewards...")

        reward_keys: List[str] = (
            self.reward_relabeling_keys
            if hasattr(self, "reward_relabeling_keys") and self.reward_relabeling_keys is not None
            else []
        )
        if len(reward_keys) == 0:
            raise RuntimeError("reward_relabeling_keys must be non-empty to relabel rewards")

        # precomputed_video_embeddings are stored concatenated across keys: [T, D_total]
        any_demo_key = next(iter(self.hdf5_cache.keys()))
        total_emb_dim = int(self.precomputed_video_embeddings[any_demo_key].shape[-1])
        if total_emb_dim % len(reward_keys) != 0:
            raise RuntimeError(
                f"precomputed_video_embeddings last dim ({total_emb_dim}) is not divisible by "
                f"len(reward_relabeling_keys) ({len(reward_keys)})."
            )
        emb_dim_per_key = total_emb_dim // len(reward_keys)

        all_rewards_flat: Dict[str, List[float]] = {key: [] for key in reward_keys}
        all_success_probs_flat: Dict[str, List[float]] = {key: [] for key in reward_keys}
        demo_to_indices: Dict[str, tuple] = {}

        for key_idx, key in enumerate(reward_keys):
            # Load frames for this specific key
            all_traj_frames, _, _ = self._load_all_frames_for_demos(key=key)
            video_frames_by_demo = {demo_key: video_frames for demo_key, _, video_frames in all_traj_frames}

            all_raw_data_flat = []
            current_idx = 0

            for demo_key, cached_demo in self.hdf5_cache.items():
                h5_path, unique_episode_id = demo_key.split("::")
                episode_len = len(cached_demo["actions"])

                # Language instruction
                language_instruction = self._load_language_instruction_from_file(
                    h5_path,
                    cached_demo.get("original_demo_name", unique_episode_id.split("_", 1)[-1]),
                )
                text_emb = self.text_embeddings_dict[language_instruction]

                # Slice embeddings chunk corresponding to this key: [T, D_per_key]
                video_embeddings_all = self.precomputed_video_embeddings[demo_key]
                start_d = key_idx * emb_dim_per_key
                end_d = (key_idx + 1) * emb_dim_per_key
                video_embeddings = video_embeddings_all[:, start_d:end_d]

                video_frames = video_frames_by_demo[demo_key]

                if self.use_eval_server:
                    # If using eval server, append only full episode (as one item) to all_raw_data_flat
                    all_raw_data_flat.append(
                        dict(
                            frames=video_frames[:episode_len],
                            task=language_instruction,
                            id=unique_episode_id,
                            metadata=dict(subsequence_length=episode_len),
                            video_embeddings=video_embeddings[:episode_len],
                            text_embedding=text_emb,
                        )
                    )
                else:
                    # Record mapping indices once (must be identical across keys)
                    if key_idx == 0:
                        demo_to_indices[demo_key] = (current_idx, current_idx + episode_len)

                    for t in range(episode_len):
                        subseq_len = t + 1
                        all_raw_data_flat.append(
                            dict(
                                frames=video_frames[:subseq_len],
                                task=language_instruction,
                                id=unique_episode_id,
                                metadata=dict(subsequence_length=subseq_len),
                                video_embeddings=video_embeddings[:subseq_len],
                                text_embedding=text_emb,
                            )
                        )

                    current_idx += episode_len

            if verbose:
                logger.info(f"Computing rewards for all demos (key={key})...")

            effective_batch_size = batch_size if batch_size is not None and batch_size > 0 else 1024
            for batch_start in tqdm(
                range(0, len(all_raw_data_flat), effective_batch_size),
                desc=f"Computing rewards ({key})",
            ):
                batch_end = min(batch_start + effective_batch_size, len(all_raw_data_flat))
                batch_raw = all_raw_data_flat[batch_start:batch_end]
                rewards_batch, success_probs_batch = self._compute_rewards_batch(batch_raw)
                all_rewards_flat[key].extend(rewards_batch)
                all_success_probs_flat[key].extend(success_probs_batch)

        # Map rewards back to demos and average across keys per timestep
        demo_idx = 0
        for demo_key, cached_demo in self.hdf5_cache.items():
            per_key_rewards = []
            per_key_success_probs = []
            if self.use_eval_server:
                for key in reward_keys:
                    per_key_rewards.append(all_rewards_flat[key][demo_idx])
                    per_key_success_probs.append(all_success_probs_flat[key][demo_idx])
            else:
                start_idx, end_idx = demo_to_indices[demo_key]
                for key in reward_keys:
                    per_key_rewards.append(all_rewards_flat[key][start_idx:end_idx])
                    per_key_success_probs.append(all_success_probs_flat[key][start_idx:end_idx])
            demo_rewards = np.array(per_key_rewards, dtype=np.float32).mean(axis=0)
            if self.add_estimated_reward:
                cached_demo["rewards"] += demo_rewards
            else:
                cached_demo["rewards"] = demo_rewards
            if self.use_success_detection:
                # Majority voting for success probabilities
                demo_dones = np.zeros_like(per_key_success_probs[0], dtype=bool)
                window = self.success_detection_duration
                threshold = float(self.success_detection_threshold)
                for t in range(len(per_key_success_probs[0]) - window + 1):
                    # Gather success probabilities for all keys in the window [t:t+window]
                    votes = 0
                    total = 0
                    for key_probs in per_key_success_probs:
                        for i in range(window):
                            total += 1
                            if key_probs[t + i] > threshold:
                                votes += 1
                    # Majority voting: if more than half are successful, mark done
                    if votes > (total // 2):
                        demo_dones[t + window - 1] = True
                cached_demo["dones"] = demo_dones

            demo_idx += 1
            assert len(cached_demo["rewards"]) == len(cached_demo["actions"]) == len(cached_demo["dones"])

        logger.info(f"Reward relabeling complete!")

    def _compute_rewards_batch(self, batch_raw: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
        """
        Compute rewards for a batch of samples using either local reward model or eval_server.

        Args:
            batch_raw: List of dictionaries, each containing frames, task, video_embeddings, text_embedding, etc.

        Returns:
            Tuple of (List of reward values as floats, List of success probabilities as floats)
        """
        if self.reward_model is not None:
            # Use local reward model
            samples = [
                raw_dict_to_sample(
                    raw_data=raw_data_item,
                    max_frames=self.max_frames,
                    sample_type="progress",
                )
                for raw_data_item in batch_raw
            ]

            is_discrete_mode = self.reward_model_config.loss.progress_loss_type == "discrete"
            progress_discrete_bins = self.reward_model_config.loss.progress_discrete_bins
            outputs = process_batch_helper(
                model_type=self.reward_model_config.model.model_type,
                model=self.reward_model,
                tokenizer=self.tokenizer,
                batch_collator=self.batch_collator,
                device=self.reward_model.device,
                batch_data=[sample.model_dump() for sample in samples],
                job_id=0,
                is_discrete_mode=is_discrete_mode,
                num_bins=progress_discrete_bins,
            )
            rewards_batch = extract_rewards_from_output(outputs)
            success_probs_batch = extract_success_probs_from_output(outputs)
        elif self.use_eval_server:
            # Use eval_server
            samples = [
                raw_dict_to_sample(
                    raw_data=raw_data_item,
                    max_frames=self.max_frames,
                    sample_type="progress",
                )
                for raw_data_item in batch_raw
            ]

            files, sample_data = build_payload(samples)
            outputs = post_batch_npy(self.eval_server_url, files, sample_data, timeout_s=self.eval_server_timeout)
            rewards_batch, success_probs_batch = extract_rewards_from_server_output(outputs)
        else:
            raise ValueError("Neither reward_model nor use_eval_server is set")

        return rewards_batch.tolist(), success_probs_batch.tolist()
