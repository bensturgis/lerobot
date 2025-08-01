#!/usr/bin/env python
"""
Visualize the ensembling uncertainty estimation method by creating a vector field 
plot of the scorer model overlaid by the actions from the sampler model.

Usage example:

Create the ensembling uncertainty estimation visualization for the Push-T task.

```
python lerobot/scripts/flow_matching/visualize_composed_seq_uncertainty_estimation.py \
    --policy.path=outputs/train/flow_matching_pusht/checkpoints/last/pretrained_model \
    --policy.device=cuda \
    --env.type=pusht
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
from lerobot.configs.visualize_ensemble import VisualizeEnsemblePipelineConfig
from lerobot.common.envs.factory import make_single_env
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.flow_matching.estimate_uncertainty import CrossEnsembleSampler
from lerobot.common.policies.flow_matching.visualizer import (
    ActionSeqVisualizer,
    VectorFieldVisualizer
)
from lerobot.common.utils.io_utils import write_video
from lerobot.common.utils.live_window import LiveWindow
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import get_safe_torch_device, init_logging

@parser.wrap()
def main(cfg: VisualizeEnsemblePipelineConfig): 
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
    policy = make_policy(
        cfg.policy,
        env_cfg=cfg.env,
        uncertainty_sampler_cfg=cfg.uncertainty_sampler
    ).to(device)
    policy.eval()

    # Initialize the cross ensemble uncertainty sampler
    policy._init_uncertainty_sampler()

    # Get cross ensemble uncertainty sampler
    cross_ensemble_sampler = policy.uncertainty_sampler
    scorer_flow_matching_model = policy.scorer
    policy.uncertainty_sampler = None
    policy.scorer = None

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
        
        # Prepare visualizers
        sampler_action_seq_visualizer = ActionSeqVisualizer(
            cfg.action_seq,
            flow_matching_cfg=policy.config,
            velocity_model=policy.flow_matching.unet,
            unnormalize_outputs=policy.unnormalize_outputs,
            output_root=ep_dir,
        )
        scorer_action_seq_visualizer = ActionSeqVisualizer(
            cfg.action_seq,
            flow_matching_cfg=policy.config,
            velocity_model=scorer_flow_matching_model.unet,
            unnormalize_outputs=policy.unnormalize_outputs,
            output_root=ep_dir,
        )
        vector_field_visualizer = VectorFieldVisualizer(
            cfg=cfg.vector_field,
            flow_matching_cfg=policy.config,
            velocity_model=scorer_flow_matching_model.unet,
            output_root=ep_dir,
        )

        # Initialize the dictionary of actions to visualize in the vector field plot
        action_data: Dict[str, Tensor] = {}

        # Roll through one episode
        max_episode_steps = env.spec.max_episode_steps
        max_vis_steps = (max_episode_steps if cfg.vis.max_steps is None
                        else min(max_episode_steps, cfg.vis.max_steps))

        start_time = time.time() 

        action_generation_iter = 0
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
                action_generation_iter += 1
                tqdm.write(f"-----------------------Action Generation Iteration {action_generation_iter}---------------------")
                
                # Stack the history of observations
                obs_batch = {
                    k: torch.stack(list(policy._queues[k]), dim=1)
                    for k in policy._queues
                    if k != "action"
                }

                # Build global-conditioning with the policy's helper
                sampler_global_cond = policy.flow_matching.prepare_global_conditioning(obs_batch)
                scorer_global_cond = scorer_flow_matching_model.prepare_global_conditioning(obs_batch)

                # Visualize action sequence batch of sampler and scorer model
                sampler_action_seq_visualizer.visualize(
                    global_cond=sampler_global_cond, env=env, dir_name="sampler_action_seq", generator=generator
                )
                scorer_action_seq_visualizer.visualize(
                    global_cond=scorer_global_cond, env=env, dir_name="scorer_action_seq", generator=generator
                )
                
                # Sample actions and get their uncertainties based on the scorer model
                sampler_actions, uncertainties = cross_ensemble_sampler.conditional_sample_with_uncertainty(
                    observation=obs_batch, generator=generator
                )
                tqdm.write(f"Cross ensemble sampler uncertainty scores: {uncertainties}")
                mean_uncertainty = float(uncertainties.mean().item())

                # Sample actions with the scorer model to compare with the sampler actions
                num_samples = cfg.ensemble_sampler.num_action_seq_samples
                scorer_actions = scorer_flow_matching_model.conditional_sample(
                    batch_size=num_samples, global_cond=scorer_global_cond.repeat(num_samples, 1), generator=generator
                )

                # Store the action samples to overlay them in the vector field plot
                action_data["scorer_actions"] = scorer_actions
                action_data["sampler_actions"] = sampler_actions[1:]
                
                # Choose an action which will be used to generate the vector field plot
                action_data["base_action"] = sampler_actions[0].unsqueeze(0)                
                
                # Visualize scorer vector field with sampler action sequences
                vector_field_visualizer.visualize(
                    global_cond=scorer_global_cond,
                    visualize_actions=True,
                    actions=action_data,
                    mean_uncertainty=mean_uncertainty,
                    generator=generator
                )

                

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