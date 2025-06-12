#!/usr/bin/env python
"""
Visualize different aspects of a Flow Matching policy rollout such as 
flow trajectories, vector fields, and generated action sequence batches.

Usage example:

Stream the policy rollout live and create flow trajectory and generated action sequence
batch visualizations.

```
python lerobot/scripts/visualize_flow_matching.py \
    --policy.path=outputs/train/flow_matching_pusht/checkpoints/last/pretrained_model \
    --policy.device=cuda \
    --env.type=pusht \
    --vis.vis_types='["flows", "action_seq"]' \
    --show=true
``` 
"""
import copy
import gymnasium as gym
import logging
import numpy as np
import random
import time
import torch
from laplace import Laplace
from laplace.baselaplace import BaseLaplace
from pathlib import Path
from tqdm import trange
from torch import nn
from torch.nn.utils import vector_to_parameters
from typing import Any, Dict, List, Optional, Tuple

from lerobot.configs import parser
from lerobot.configs.visualize_laplace import VisualizeLaplacePipelineConfig
from lerobot.common.policies.factory import make_policy, make_flow_matching_visualizers
from lerobot.common.policies.flow_matching.uncertainty_estimation_utils import (
    create_laplace_calibration_dataloader,
    PointwiseConv1dToLinear,
    VelocityModelWrapper,
)
from lerobot.common.envs.factory import make_single_env
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.utils.io_utils import write_video
from lerobot.common.utils.live_window import LiveWindow
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import get_safe_torch_device, init_logging

def capture_pusht_state(env: gym.Env) -> Dict[str, np.ndarray]:
    """Get state of Push-T env as [agent_x, agent_y, block_x, block_y, block_angle]
    as well as other info."""
    last_action = env.unwrapped._last_action if env.unwrapped._last_action is not None else None
    return {
        "state": np.array([
            *env.unwrapped.agent.position,
            *env.unwrapped.block.position,
            env.unwrapped.block.angle,
        ], dtype=np.float64),
        "agent_velocity": np.array([*env.unwrapped.agent.velocity]),
        "last_action": last_action
    }

def restore_pusht_state(env: gym.Env, info: Dict[str, np.ndarray]) -> np.ndarray:
    """Place PushT into the given state."""
    state = info["state"]
    env.unwrapped.agent.position = list(state[:2])
    env.unwrapped.block.position = list(state[2:4])
    env.unwrapped.block.angle = state[4]
    agent_velocity = list(info["agent_velocity"])
    env.unwrapped.agent.velocity = agent_velocity
    last_action = info["last_action"]
    env.unwrapped._last_action = last_action
    env.unwrapped.space.step(0.001)

    return env.unwrapped.get_obs()

def draw_laplace_velocity_model(
    laplace_posterior: BaseLaplace,
    wrapped_velocity_model: nn.Module,
) -> nn.Module:
    """
    Returns a fresh copy of the MAP velocity model whose final linear
    layer weights are drawn from the Laplace posterior.
    """
    laplace_model_weights = laplace_posterior.sample(n_samples=1).squeeze(0)
    wrapped_laplace_velocity_model = copy.deepcopy(wrapped_velocity_model)
    last_linear_layer_params = list(
        wrapped_laplace_velocity_model.base_model.final_conv[1].parameters()
    )
    vector_to_parameters(laplace_model_weights, last_linear_layer_params)
    laplace_velocity_model = wrapped_laplace_velocity_model.base_model
    laplace_velocity_model.eval()

    return laplace_velocity_model.to(laplace_model_weights.device)

def rollout(
    cfg: VisualizeLaplacePipelineConfig,
    env: gym.Env,
    policy,
    output_dir: Path,
    seed: int,
    env_states: Optional[List[Any]] = None
) -> list[Any]:
    device = get_safe_torch_device(cfg.policy.device, log=True)
    
    observation, _ = env.reset(seed=seed)
    env.unwrapped._last_action = None
    
    # Callback for visualization.
    def render_frame(env: gym.Env) -> np.ndarray:
        rgb_frame = env.render()
        video_frames.append(rgb_frame)

        # Live visualization
        if cfg.show:
            live_view.enqueue_frame(rgb_frame[..., ::-1])
        
        return rgb_frame
    
    # Cache frames for creating video
    video_frames: list[np.ndarray] = []
    # Chache environment states
    state_history: list[Any] = []
    # Setup for live visualization
    if env_states is None:
        if cfg.show:
            live_view = LiveWindow("Live Visualization")
        render_frame(env)

    # Prepare visualisers
    visualizers = make_flow_matching_visualizers(
        vis_cfg=cfg.vis,
        model_cfg=policy.config,
        velocity_model=policy.flow_matching.unet,
        output_root=output_dir,
        unnormalize_outputs=policy.unnormalize_outputs,
    )

    # Roll through one episode
    max_episode_steps = env.spec.max_episode_steps
    max_vis_steps = (max_episode_steps if cfg.vis.max_steps is None
                    else min(max_episode_steps, cfg.vis.max_steps))

    start_time = time.time() 

    progbar = trange(
        max_vis_steps,
        desc=f"Running rollout with at most {max_vis_steps} steps"
    )   
    for step_idx in progbar:
        if env_states is None:
            state_history.append(capture_pusht_state(env))
        
        if env_states is not None:
            if step_idx >= len(env_states):
                break
            if step_idx > 0:
                observation = restore_pusht_state(env, env_states[step_idx])
        
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
        observation = preprocess_observation(observation)
        observation = {
            key: observation[key].to(device, non_blocking=device.type == "cuda") for key in observation
        }
        
        # Decide whether a new sequence will be generated
        new_action_gen = len(policy._queues["action"]) == 0       

        with torch.no_grad():
            action = policy.select_action(observation)

        if new_action_gen and (cfg.vis.start_step is None or step_idx >= cfg.vis.start_step):
            # Stack the history of observations
            batch = {
                k: torch.stack(list(policy._queues[k]), dim=1)
                for k in policy._queues
                if k != "action"
            }

            # build global-conditioning with the policy's helper
            global_cond = policy.flow_matching.prepare_global_conditioning(batch)

            for visualizer in visualizers:
                visualizer.visualize(global_cond=global_cond, env=env)                

        # Apply the next action
        observation, _, terminated, _, _ = env.step(action[0].cpu().numpy())
        if env_states is None:
            render_frame(env)

        # Stop early if environment terminates
        if terminated:
            break

    logging.info(f"Finished in {time.time() - start_time:.1f}s")

    # Close the live visualization
    if cfg.show:
        live_view.close()

    # Save the buffered video
    if env_states is None:
        write_video(
            str(output_dir / "rollout.mp4"),
            np.stack(video_frames, axis=0),
            fps=env.metadata["render_fps"]
        )

    return state_history

@parser.wrap()
def main(cfg: VisualizeLaplacePipelineConfig): 
    # Set global seed
    if cfg.seed is not None:
        set_seed(cfg.seed)
    
    logging.info("Loading policy")
    if cfg.policy.type != "flow_matching":
        raise ValueError(
            f"visualize_flow_matching.py only supports Flow Matching policies, "
            f"but got policy type '{cfg.policy.type}'."
        )
    device = get_safe_torch_device(cfg.policy.device, log=True)
    policy = make_policy(cfg.policy, env_cfg=cfg.env).to(device)
    policy.eval()

    logging.info("Creating environment")
    env = make_single_env(cfg.env)
    # ------------------------------------------
    velocity_model = policy.flow_matching.unet
    velocity_model.final_conv[1] = PointwiseConv1dToLinear(velocity_model.final_conv[1])
    wrapped_velocity_model = VelocityModelWrapper(velocity_model)
    laplace_posterior = Laplace(
        wrapped_velocity_model,
        likelihood="regression",
        subset_of_weights="last_layer",
        hessian_structure="diag"
    )
    laplace_calib_loader = create_laplace_calibration_dataloader(
        cfg=cfg,
        policy=policy,
    )
    laplace_posterior.fit(laplace_calib_loader)
    laplace_velocity_models = []
    for k in range(cfg.n_laplace_models):
        laplace_velocity_models.append(draw_laplace_velocity_model(
            laplace_posterior=laplace_posterior,
            wrapped_velocity_model=wrapped_velocity_model,
        ))
    # ------------------------------------------
    seed = cfg.seed if cfg.seed is not None else random.randrange(2**32)
    state_history = rollout(
        cfg=cfg,
        env=env,
        policy=policy,
        seed=seed,
        output_dir=cfg.output_dir,
    )

    for k, laplace_velocity_model in enumerate(laplace_velocity_models):
        policy.reset()
        # Replace flow matching MAP velocity model with a Laplace approximation
        policy.flow_matching.unet = laplace_velocity_model
        output_dir = cfg.output_dir / f"laplace_model_{k+1}"

        rollout(
            cfg=cfg,
            env=env,
            policy=policy,
            output_dir=output_dir,
            seed=seed,
            env_states=state_history,
        )


if __name__ == "__main__":
    init_logging()
    main()