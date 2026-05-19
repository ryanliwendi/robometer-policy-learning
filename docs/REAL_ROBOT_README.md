# Real Robot Training with DSRL

**See also:** [robometer_policy_learning/robots/README.md](robometer_policy_learning/robots/README.md) (canonical DROID + WidowX TCP servers, shared protocol, Pinggy workflow). [README — Remote Reward Relabeling](README.md#remote-reward-relabeling) for async Robometer/RoboReward gRPC. [README — Remote Robot Training](README.md#remote-robot-training) for a doc map.

This guide explains how to train DSRL on a real robot using the remote robot server architecture. This setup works across different machines using TCP tunneling services like **Pinggy** or **ngrok**.

## Architecture Overview

```
┌─────────────────────────────────────────┐     ┌─────────────────────────────────────┐
│         Robot Machine                   │     │       Training Machine              │
│                                         │     │                                     │
│  ┌──────────────────┐                   │     │   ┌─────────────────────────────┐   │
│  │  WidowX Robot    │                   │     │   │    train_dsrl.py            │   │
│  └────────┬─────────┘                   │     │   │    (SAC + Pi0)              │   │
│           │                             │     │   └──────────────┬──────────────┘   │
│  ┌────────▼─────────┐    TCP/6000       │     │                  │                  │
│  │ widowx_remote_   │◄──────────────────┼─────┼──────────────────┘                  │
│  │ server.py        │    (via tunnel)   │     │   RemoteRobotEnv connects to        │
│  └──────────────────┘                   │     │   robot server over TCP             │
│                                         │     │                                     │
│  Keyboard: s=success, f=failure, q=quit │     │                                     │
└─────────────────────────────────────────┘     └─────────────────────────────────────┘
```

## WidowX controller (BRIDGE stack)

If you use the **WidowX environment service** from the [BRIDGE / `bridge_data_robot`](https://github.com/rail-berkeley/bridge_data_robot) stack (same stack as BRIDGE data collection), set that up before the Quick Start below.

1. **Install** WidowX-related dependencies for this repo (example with `uv`):
   ```bash
   uv pip install -e ".[widowx]"
   ```
   Use `pip install -e '.[widowx]'` if you do not use `uv`.

2. **Robot hardware**: follow the BRIDGE repository instructions for the physical WidowX setup.

3. **Environment service** (typical flow from the `bridge_data_robot` checkout):
   ```bash
   USB_CONNECTOR_CHART=$(pwd)/usb_connector_chart.yml docker compose up --build robonet
   # In a separate terminal:
   docker compose exec robonet bash -lic "widowx_env_service --server"
   ```

`widowx_remote_server.py` connects to the controller at `--robot-ip` / `--robot-port` (defaults target the service above).

## Quick Start

### 1. Start the Robot Server (Robot Machine)

```bash
cd /path/to/rfm_rl

uv run python robometer_policy_learning/robots/widowx_remote_server.py \
    --robot-ip localhost \
    --robot-port 5556 \
    --server-port 6000 \
    --prompt "pick up the red block and place it in the bowl" \
    --max-steps 60
```

Then open a **second** terminal for tunneling (see §2).

**Arguments:**
- `--robot-ip`: IP of the WidowX controller (default: `localhost`)
- `--robot-port`: Port of the WidowX controller (default: `5556`)
- `--server-port`: Port for the remote server (default: `6000`)
- `--prompt`: Task instruction for the robot (**required**)
- `--max-steps`: Max steps per episode before timeout (default: `120`)
- `--resolution`: Image resolution (default: `224`)
- `--wait-for-enter`: Wait for Enter before each episode (default: `True`)
- `--no-wait-for-enter`: Start episodes immediately without waiting

### 2. Set Up Tunneling (Robot Machine)

#### Option A: Pinggy (Free, Recommended)
```bash
ssh -p 443 -R0:localhost:6000 qr+tcp@free.pinggy.io
```
Note the URL provided (e.g., `abc123.a.pinggy.io`). The port is `443`.

#### Option B: ngrok
```bash
ngrok tcp 6000
```
Note the URL provided (e.g., `0.tcp.ngrok.io:12345`).

### 3. Start Training (Training Machine)

```bash

# make sure you have the Pi0 checkpoint downloaded
uv run hf download jesbu1/pi0_lora_bridge_1_cam --local-dir pi0_lora_bridge_1_cam

uv run python scripts/train_dsrl.py \
    config_name=dsrl_bridge_config.yaml \
    remote_robot.host=TCP_LINK \
    remote_robot.port=TCP_PORT \
    pi0_checkpoint=./pi0_lora_bridge_1_cam/pi0_lora_bridge_1_cam/29999 \
    num_rollouts=10000 \
    eval_freq=1000
```

### 4. Optional: DSRL + async reward relabeling (DROID)

For DROID-style remote environments with **Robometer** or **RoboReward** scoring trajectories over gRPC, start the reward relabel server then `train_dsrl.py` with `dsrl_remote_robot_async_relabel_config`. Full copy-paste commands, success-threshold examples, multistage variant, and **eval** (`eval_trained_dsrl.py`) are in the main [README — Remote Reward Relabeling](README.md#remote-reward-relabeling).

## Episode Workflow

By default, the robot waits for you to press **Enter** before starting each episode:

```
============================================================
Resetting robot for new episode...
============================================================

Task: pick up the red block
Max steps: 100

────────────────────────────────────────────────────────────
🤖 Robot is ready. Set up the scene if needed.
────────────────────────────────────────────────────────────
>>> Press ENTER to start episode...
```

This gives you time to:
- Position objects in the scene
- Move obstacles out of the way
- Ensure the robot's workspace is clear

To disable this and start episodes immediately, use `--no-wait-for-enter`.

## Keyboard Controls

While the robot server is running, you can use keyboard controls to mark episodes:

| Key | Action |
|-----|--------|
| `ENTER` | Start episode after reset |
| `s` | Mark current episode as **SUCCESS** (reward = 1) |
| `f` | Mark current episode as **FAILURE** (reward = 0) |
| `q` | Quit server |

**Note:** Press the key directly (no Enter required on Linux/Mac).

## Important Notes

### Evaluation at Episode Boundaries

For real robots, evaluation **only occurs when an episode completes** (not mid-trajectory). This is automatically enabled when `env_name=REMOTE_ROBOT` because:
- The training and eval environments are the same physical robot
- Interrupting a trajectory could leave the robot in an unsafe state

### Reward Structure

- **Success (operator presses 's'):** reward = 1.0 → transformed to 0.0 for DSRL
- **Failure/Ongoing:** reward = 0.0 → transformed to -1.0 for DSRL
- **Timeout (max_steps reached):** episode truncated, reward = 0.0

### Connection Handling

The client automatically:
- Retries connection for up to 5 minutes (configurable)
- Reconnects if connection is lost during training
- Handles network interruptions gracefully

## Troubleshooting

### Connection Refused
```
Connection failed: [Errno 111] Connection refused
```
- Ensure the robot server is running
- Check that the tunnel is active
- Verify host/port are correct

### Timeout During Training
```
Failed to connect after 300s
```
- Increase `remote_robot.connect_timeout`
- Check network connectivity
- Restart the tunnel

### Robot Not Responding
- Check the robot controller is running (`--robot-ip`, `--robot-port`)
- Look for errors in the robot server terminal
- Try restarting the robot server
- If you use the BRIDGE docker stack, confirm `widowx_env_service` is running inside `robonet`

### Image or camera issues
- Verify camera settings in `WidowXConfigs.DefaultEnvParams` (or your deployment’s equivalent)
- Check camera permissions and USB connections
- Ensure `--resolution` matches what the policy expects (default `224`)

## Safety

- Supervise the robot whenever the server is running
- Keep an emergency stop within reach
- Start with small motions and clear the workspace before long runs
- Prefer a dry run in a constrained workspace before long training sessions

## WidowX observation layout (server-side)

The WidowX remote server processes end-effector state and images before they are sent over TCP:

1. Quaternion rotation is converted to Euler angles, then transformed to a top-down frame consistent with BRIDGE conventions.
2. Returned proprio state layout: `[x, y, z, roll, pitch, yaw, gripper_openness]`.
3. RGB images are resized to the requested resolution (default 224×224) from the camera configured for your WidowX deployment.

This matches the Pi0 / BRIDGE-style layout consumed by `dsrl_bridge_config` training.

## Protocol Reference

Length-prefixed **pickle** over TCP matches the socket servers documented in [robometer_policy_learning/robots/README.md](robometer_policy_learning/robots/README.md) (**Communication Protocol**). The snippets below focus on the **WidowX** observation payload shape for this bridge setup.

**RESET:**
```python
# Client sends:
{'type': 'RESET'}

# Server responds:
{
    'state': np.ndarray,      # [x, y, z, roll, pitch, yaw, gripper] (7,)
    'image': np.ndarray,      # RGB image (224, 224, 3)
    'instruction': str        # Task prompt
}
```

**STEP:**
```python
# Client sends:
{'type': 'STEP', 'action': np.ndarray}  # (7,) action

# Server responds:
{
    'state': np.ndarray,
    'image': np.ndarray,
    'instruction': str,
    'reward': float,          # 0.0 or 1.0
    'done': bool,
    'truncated': bool,
    'info': dict
}
```

**CLOSE:**
```python
{'type': 'CLOSE'}  # Server closes connection
```

