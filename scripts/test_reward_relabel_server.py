#!/usr/bin/env python3
"""
Simple test script for the reward relabeling server.

Usage:
    # Test with default server address (localhost:50052):
    uv run python scripts/test_reward_relabel_server.py

    # Test with custom server address:
    uv run python scripts/test_reward_relabel_server.py --server-address localhost:50052 --num-steps 5
"""

import argparse
import numpy as np
import grpc
from robometer_policy_learning.distributed.protos import reward_relabel_pb2 as pb
from robometer_policy_learning.distributed.protos import reward_relabel_pb2_grpc as pb_grpc
from robometer_policy_learning.distributed.protos import learner_pb2 as learner_pb
from robometer_policy_learning.distributed.grpc_utils import ndarray_to_bytes


def create_mock_transition(step: int, num_frames: int = 4, image_shape=(224, 224, 3), dino_dim=768, text_dim=384):
    """Create a mock transition with dummy data for testing."""
    # Create dummy image frame (H, W, C)
    image = np.random.randint(0, 255, size=image_shape, dtype=np.uint8)

    # Create dummy DINO embedding
    dino_embedding = np.random.randn(dino_dim).astype(np.float32)

    # Create dummy text embedding (same for all transitions in episode)
    text_embedding = np.random.randn(text_dim).astype(np.float32)

    # Create dummy action
    action = np.random.randn(4).astype(np.float32)

    # Build observation dict
    obs = {
        "observation/image": image,
        # "dino_embedding": dino_embedding,
        # "language": text_embedding,
    }

    # Build next observation (same structure)
    next_obs = {
        "observation/image": np.random.randint(0, 255, size=image_shape, dtype=np.uint8),
        # "dino_embedding": np.random.randn(dino_dim).astype(np.float32),
        # "language": text_embedding,  # Same text embedding
    }

    return obs, action, next_obs


def test_reward_relabel_server(server_address: str = "localhost:50052", num_steps: int = 5):
    """Test the reward relabeling server with mock transitions."""
    print(f"Connecting to reward relabeling server at {server_address}...")

    # Create gRPC channel and stub
    channel = grpc.insecure_channel(server_address)
    stub = pb_grpc.RewardRelabelServiceStub(channel)

    try:
        # Test connection
        grpc.channel_ready_future(channel).result(timeout=5)
        print("✓ Connected to server")
    except grpc.FutureTimeoutError:
        print(f"✗ Failed to connect to server at {server_address}")
        print("  Make sure the server is running with:")
        print("    python scripts/start_reward_relabel_server.py")
        return

    # Create mock trajectory
    print(f"\nCreating mock trajectory with {num_steps} transitions...")
    transitions = []
    language_instructions = []
    episode_ids = []
    step_in_episodes = []

    episode_id = "test_episode_001"
    language_instruction = "pick up the red block"

    for step in range(num_steps):
        obs, action, next_obs = create_mock_transition(step)

        # Convert to protobuf format
        obs_proto = {k: learner_pb.NDArray(data=ndarray_to_bytes(v)) for k, v in obs.items()}
        next_obs_proto = {k: learner_pb.NDArray(data=ndarray_to_bytes(v)) for k, v in next_obs.items()}
        action_proto = learner_pb.NDArray(data=ndarray_to_bytes(action))

        tr = learner_pb.Transition(
            obs=obs_proto,
            action=action_proto,
            reward_env=0.0,  # Dummy reward
            next_obs=next_obs_proto,
            done=(step == num_steps - 1),  # Last step is done
            truncated=False,
            episode_id=episode_id,
            step_in_episode=step,
            timestamp_ns=0,
        )
        transitions.append(tr)
        language_instructions.append(language_instruction)
        episode_ids.append(episode_id)
        step_in_episodes.append(step)

    # Create request
    request = pb.RelabelRewardsRequest(
        transitions=transitions,
        language_instructions=language_instructions,
        episode_ids=episode_ids,
        step_in_episodes=step_in_episodes,
    )

    print(f"Sending request to server...")
    try:
        # Send request
        response = stub.RelabelRewards(request, timeout=60)

        if response.ok:
            print(f"✓ Server responded successfully")
            print(f"  Message: {response.message}")
            print(f"  Number of rewards: {len(response.rewards)}")
            print(f"  Rewards: {[f'{r:.4f}' for r in response.rewards]}")

            # Verify we got the right number of rewards
            if len(response.rewards) == num_steps:
                print(f"✓ Received correct number of rewards ({num_steps})")
            else:
                print(f"✗ Expected {num_steps} rewards, got {len(response.rewards)}")
        else:
            print(f"✗ Server returned error: {response.message}")

    except grpc.RpcError as e:
        print(f"✗ gRPC error: {e.code()}: {e.details()}")
    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        channel.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test reward relabeling server")
    parser.add_argument(
        "--server-address",
        type=str,
        default="localhost:50052",
        help="Server address (default: localhost:50052)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=5,
        help="Number of transitions to send (default: 5)",
    )

    args = parser.parse_args()

    test_reward_relabel_server(args.server_address, args.num_steps)
