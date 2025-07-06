#!/usr/bin/env python
"""
Visualize the composed action sequence uncertainty estimation method by creating a vector field
plot overlaid by the composed action sequences.

Usage example:

Stream the policy rollout live and create the composed action sequence visualization for the Push-T
task.

```
python lerobot/scripts/flow_matching/visualize_composed_seq_uncertainty_estimation.py \
    --policy.path=outputs/train/flow_matching_pusht/checkpoints/last/pretrained_model \
    --policy.device=cuda \
    --env.type=pusht \
    --show=true
``` 
"""
import gymnasium as gym
import logging
import numpy as np
import time
import torch

from torch import Tensor
from tqdm import trange, tqdm
from typing import Dict

from lerobot.configs import parser
from lerobot.configs.visualize_composed_seq import VisualizeComposedSeqPipelineConfig
from lerobot.common.envs.factory import make_single_env
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.flow_matching.estimate_uncertainty import ComposedSequenceSampler
from lerobot.common.policies.flow_matching.visualizer import VectorFieldVisualizer
from lerobot.common.utils.io_utils import write_video
from lerobot.common.utils.live_window import LiveWindow
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import get_safe_torch_device, init_logging

@parser.wrap()
def main(cfg: VisualizeComposedSeqPipelineConfig): 
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

    num_rollouts = cfg.vis.num_rollouts
    for ep in range(num_rollouts):
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
        render_frame(env)

        ep_dir = cfg.output_dir / f"rollout_{ep:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        # Prepare composed action sequence uncertainty sampler
        composed_seq_sampler = ComposedSequenceSampler(
            flow_matching_cfg=policy.config, 
            cfg=cfg.composed_seq_sampler,
            velocity_model=policy.flow_matching.unet
        )
        
        # Prepare visualizers
        vector_field_visualizer = VectorFieldVisualizer(
            cfg=cfg.vector_field,
            flow_matching_cfg=policy.config,
            velocity_model=policy.flow_matching.unet,
            output_root=ep_dir,
        )

        # Only visualize the action steps that come from the next observation
        prev_action_seq_end = (
            policy.config.n_obs_steps - 1 + policy.config.n_action_steps
        )
        horizon = policy.config.horizon
        attached_action_steps = list(range(prev_action_seq_end, horizon))
        if vector_field_visualizer.action_steps not in attached_action_steps:
            vector_field_visualizer.action_steps = list(range(prev_action_seq_end, horizon))

        # Initialize the dictionary of actions to visualize in the vector field plot
        action_data: Dict[str, Tensor] = {}

        # At the beginning of an epsiode we don't have previous actions to compose with
        prev_global_cond: Tensor | None = None
        prev_actions: Tensor | None = None
        composed_seq_sampler.prev_action_sequence = None

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
                action = policy.select_action(observation)

            if new_action_gen and (cfg.vis.start_step is None or step_idx >= cfg.vis.start_step):
                # Stack the history of observations
                batch = {
                    k: torch.stack(list(policy._queues[k]), dim=1)
                    for k in policy._queues
                    if k != "action"
                }

                # Build global-conditioning with the policy's helper
                global_cond = policy.flow_matching.prepare_global_conditioning(batch)

                # Get the newly sampled actions
                new_actions, uncertainties = composed_seq_sampler.conditional_sample_with_uncertainty(
                    global_cond=global_cond
                )
                tqdm.write(f"Compsed sequence sampler uncertainty scores: {uncertainties}")
                mean_uncertainty = float(uncertainties.mean().item())

                # Compose actions
                if prev_actions is not None:
                    composed_actions = composed_seq_sampler.compose_action_seqs(
                        prev_action_seq=prev_actions,
                        new_action_seq=new_actions
                    )
                    action_data["action_samples"] = prev_actions
                    action_data["composed_actions"] = composed_actions

                    # Visualize vector field with composed action sequences
                    vector_field_visualizer.visualize(
                        global_cond=prev_global_cond,
                        visualize_actions=True,
                        actions=action_data,
                        mean_uncertainty=mean_uncertainty,
                    )                

                # Set the previous global conditioning vector and the previous action sequences to compose with
                prev_global_cond = global_cond
                prev_actions = new_actions
                composed_seq_sampler.prev_action_sequence = new_actions

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