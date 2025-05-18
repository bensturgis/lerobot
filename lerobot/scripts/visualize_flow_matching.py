#!/usr/bin/env python
import gymnasium as gym
import logging
import numpy as np
import time
import torch
from tqdm import trange

from lerobot.configs import parser
from lerobot.configs.visualize import VisualizePipelineConfig
from lerobot.common.policies.factory import make_policy, make_flow_matching_visualizer
from lerobot.common.envs.factory import make_single_env
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.utils.io_utils import write_video
from lerobot.common.utils.live_window import LiveWindow
from lerobot.common.utils.utils import get_safe_torch_device, init_logging

@parser.wrap()
def main(cfg: VisualizePipelineConfig): 
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
    observation, _ = env.reset(seed=cfg.seed)
    
    # Callback for visualization.
    def render_frame(env: gym.Env):
        video_frames.append(env.render())

        # Live visualization
        if cfg.show:
            rgb = env.render()
            live_view.enqueue_frame(rgb[..., ::-1])

    # Cache frames for creating video
    video_frames: list[np.ndarray] = []
    # Setup for live visualization
    if cfg.show:
        live_view = LiveWindow("Live Visualization")
    render_frame(env)

    # Prepare visualiser
    visualizer = make_flow_matching_visualizer(
        vis_cfg=cfg.vis,
        model_cfg=policy.config,
        velocity_model=policy.flow_matching.unet,
        output_root=cfg.output_dir,
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
            global_cond = policy.flow_matching._prepare_global_conditioning(batch)

            visualizer.visualize(global_cond=global_cond)            

        # Apply the next action
        observation, _, terminated, truncated, _ = env.step(action[0].cpu().numpy())
        render_frame(env)

        # Stop early if environment terminates
        done = terminated or truncated
        if done:
            break

    logging.info(f"Finished in {time.time() - start_time:.1f}s")

    # Close the live visualization
    if cfg.show:
        live_view.close()

    # Save the buffered video
    write_video(
        str(cfg.output_dir / "rollout.mp4"),
        np.stack(video_frames, axis=0),
        fps=env.metadata["render_fps"]
    )


if __name__ == "__main__":
    init_logging()
    main()