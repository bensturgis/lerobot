#!/usr/bin/env python
"""
Visualize different aspects of a Flow Matching policy rollout such as 
flow trajectories, vector fields, and generated action sequence batches.

Usage example:

Stream the policy rollout live and create flow trajectory and generated action sequence
batch visualizations.

```
python lerobot/scripts/flow_matching/visualize.py \
    --policy.path=outputs/train/flow_matching_pusht/checkpoints/last/pretrained_model \
    --policy.device=cuda \
    --env.type=pusht \
    --vis.vis_types='["flows", "action_seq"]' \
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
from lerobot.common.envs.factory import make_single_env
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.flow_matching.visualizer import (
    FlowMatchingVisualizer,
    ActionSeqVisualizer,
    FlowVisualizer,
    VectorFieldVisualizer,
)
from lerobot.common.utils.io_utils import write_video
from lerobot.common.utils.live_window import LiveWindow
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import get_safe_torch_device, init_logging

@parser.wrap()
def main(cfg: VisualizePipelineConfig): 
    # Set global seed
    if cfg.seed is not None:
        set_seed(cfg.seed)
        rollout_seeds = list(range(cfg.seed, cfg.seed + cfg.vis.num_rollouts))
    else:
        rollout_seeds = None
    
    logging.info("Loading policy")
    if cfg.policy.type != "flow_matching":
        raise ValueError(
            f"visualize_flow_matching.py only supports Flow Matching policies, "
            f"but got policy type '{cfg.policy.type}'."
        )
    device = get_safe_torch_device(cfg.policy.device, log=True)
    policy = make_policy(cfg.policy, env_cfg=cfg.env).to(device)
    policy.eval()

    num_rollouts = cfg.vis.num_rollouts
    for ep in range(num_rollouts):
        generator = torch.Generator(device=device)
        if rollout_seeds is None:
            seed = None
        else:
            seed = rollout_seeds[ep]
            generator.manual_seed(seed)

        logging.info("Creating environment")
        env = make_single_env(cfg.env, seed)
        
        reset_kwargs: dict = {}
        if cfg.start_state is not None and cfg.env.type == "pusht":
            logging.info(f"Resetting to provided start_state {cfg.start_state}")
            reset_kwargs["options"] = {"reset_to_state": cfg.start_state}

        observation, _ = env.reset(seed=seed, **reset_kwargs)
        
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
        render_frame(env)

        ep_dir = cfg.output_dir / f"rollout_{ep:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        # Prepare visualisers
        visualizers: list[FlowMatchingVisualizer] = []
        if "action_seq" in cfg.vis_types:
            visualizers.append(
                ActionSeqVisualizer(
                    cfg=cfg.action_seq,
                    flow_matching_cfg=policy.config,
                    velocity_model=policy.flow_matching.unet,
                    unnormalize_outputs=policy.unnormalize_outputs,
                    output_root=ep_dir,
                )
            )
        if "flows" in cfg.vis_types:
            visualizers.append(
                FlowVisualizer(
                    cfg=cfg.flows,
                    flow_matching_cfg=policy.config,
                    velocity_model=policy.flow_matching.unet,
                    output_root=ep_dir,
                )
            )
        if "vector_field" in cfg.vis_types:
            visualizers.append(
                VectorFieldVisualizer(
                    cfg=cfg.vector_field,
                    flow_matching_cfg=policy.config,
                    velocity_model=policy.flow_matching.unet,
                    output_root=ep_dir,
                )
            )

        # Roll through one episode
        max_episode_steps = env.spec.max_episode_steps
        max_vis_steps = (max_episode_steps if cfg.vis.max_steps is None
                        else min(max_episode_steps, cfg.vis.max_steps))

        start_time = time.time() 

        progbar = trange(
            max_vis_steps,
            desc=f"[Episode {ep+1}/{num_rollouts}]: Running rollout with at most {max_vis_steps} steps"
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
                action = policy.select_action(observation, generator)

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
                    visualizer.visualize(global_cond=global_cond, env=env, generator=generator)                

            # Apply the next action
            observation, _, terminated, _, _ = env.step(action[0].cpu().numpy())
            render_frame(env)

            # Stop early if environment terminates
            if terminated:
                break

        logging.info(f"Finished in {time.time() - start_time:.1f}s")

        env.close()

        # Close the live visualization
        if cfg.show:
            live_view.close()

        # Save the buffered video
        write_video(
            str(ep_dir / "rollout.mp4"),
            np.stack(video_frames, axis=0),
            fps=env.metadata["render_fps"]
        )


if __name__ == "__main__":
    init_logging()
    main()