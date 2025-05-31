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
    --vis.vis_types='["flows", "action_seq"]'
    --show=true
``` 
"""
import gymnasium as gym
import logging
import numpy as np
import time
import torch
from tqdm import trange

from lerobot.configs import parser
from lerobot.configs.visualize import VisualizePipelineConfig
from lerobot.common.policies.factory import make_policy, make_flow_matching_visualizers
from lerobot.common.envs.factory import make_single_env
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.utils.io_utils import write_video
from lerobot.common.utils.live_window import LiveWindow
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import get_safe_torch_device, init_logging

@parser.wrap()
def main(cfg: VisualizePipelineConfig): 
    # Set global seed
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
    
    reset_kwargs: dict = {}
    if cfg.start_state is not None and cfg.env.type == "pusht":
        logging.info(f"Resetting to provided start_state {cfg.start_state}")
        reset_kwargs["options"] = {"reset_to_state": cfg.start_state}

    observation, _ = env.reset(seed=cfg.seed, **reset_kwargs)
    
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
    # Setup for live visualization
    if cfg.show:
        live_view = LiveWindow("Live Visualization")
    cur_frame = render_frame(env)

    # Prepare visualisers
    visualizers = make_flow_matching_visualizers(
        vis_cfg=cfg.vis,
        model_cfg=policy.config,
        velocity_model=policy.flow_matching.unet,
        output_root=cfg.output_dir,
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

            for visualizer in visualizers:
                visualizer.visualize(global_cond=global_cond, frame=cur_frame.copy())                

        # Apply the next action
        observation, _, terminated, truncated, _ = env.step(action[0].cpu().numpy())
        cur_frame = render_frame(env)

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