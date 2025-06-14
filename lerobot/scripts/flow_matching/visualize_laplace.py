#!/usr/bin/env python
import gymnasium as gym
import logging
import numpy as np
import random
import time
import torch
from laplace import Laplace
from pathlib import Path
from torch import nn
from tqdm import trange
from typing import Any, Dict, List, Optional

from lerobot.configs import parser
from lerobot.configs.visualize_laplace import VisualizeLaplacePipelineConfig
from lerobot.common.policies.factory import make_policy, make_flow_matching_visualizers
from lerobot.common.policies.flow_matching.uncertainty_estimation_utils import (
    create_laplace_flow_matching_calib_loader,
    draw_laplace_flow_matching_model,
    FlowMatchingModelWrapper,
    PointwiseConv1dToLinear,
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

    return env.get_obs()

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
            f"visualize_laplace.py only supports Flow Matching policies, "
            f"but got policy type '{cfg.policy.type}'."
        )
    device = get_safe_torch_device(cfg.policy.device, log=True)
    policy = make_policy(cfg.policy, env_cfg=cfg.env).to(device)
    policy.eval()

    logging.info("Creating environment")
    env = make_single_env(cfg.env)
    # ------------------------------------------
    flow_matching_model = policy.flow_matching
    laplace_approx_targets: list[nn.Module] = []
    if cfg.laplace_scope in ["velocity_last", "both"]:
        flow_matching_model.unet.final_conv[1] = PointwiseConv1dToLinear(
            flow_matching_model.unet.final_conv[1]
        )
        laplace_approx_targets.append("unet.final_conv.1.linear_layer")

    if cfg.laplace_scope in ["rgb_last", "both"]:
        laplace_approx_targets.append("rgb_encoder.out")

    if len(laplace_approx_targets) == 0:
        raise ValueError(
            f"Unknown laplace_scope={cfg.laplace_scope}. Choose from "
            "'velocity_last', 'rgb_last' and 'both'."
        )
    
    # Freeze all parameters
    for p in flow_matching_model.parameters():
        p.requires_grad_(False)

    # Un-freeze parameters from target modules
    for name, module in flow_matching_model.named_modules():
        if name in laplace_approx_targets:
            for p in module.parameters():
                p.requires_grad_(True)

    wrapped_flow_matching_model = FlowMatchingModelWrapper(flow_matching_model)
    laplace_posterior = Laplace(
        wrapped_flow_matching_model,
        likelihood="regression",
        subset_of_weights="all",
        hessian_structure="diag",
    )
    
    laplace_calib_loader = create_laplace_flow_matching_calib_loader(
        cfg=cfg,
        policy=policy,
        calib_fraction=cfg.calib_fraction
    )
    laplace_posterior.fit(laplace_calib_loader)
    laplace_flow_matching_models = []
    for k in range(cfg.n_laplace_models):
        laplace_flow_matching_models.append(draw_laplace_flow_matching_model(
            laplace_posterior=laplace_posterior,
            flow_matching_model=wrapped_flow_matching_model.base_model,
            target_modules=laplace_approx_targets
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

    for k, laplace_flow_matching_model in enumerate(laplace_flow_matching_models):
        policy.reset()
        # Replace flow matching MAP velocity model with a Laplace approximation
        policy.flow_matching = laplace_flow_matching_model
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