#!/usr/bin/env python
import logging
import time
import torch
from tqdm import trange

from lerobot.configs import parser
from lerobot.configs.visualize import VisualizePipelineConfig
from lerobot.common.policies.factory import make_policy, make_flow_matching_visualizer
from lerobot.common.envs.factory import make_env
from lerobot.common.envs.utils import preprocess_observation
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
    env = make_env(cfg.env, n_envs=1, use_async_envs=False)   # visualise first env only
    observation, _ = env.reset(seed=cfg.seed)
    start_time = time.time()

    # Prepare visualiser
    vis = make_flow_matching_visualizer(
        vis_cfg=cfg.vis,
        model_cfg=policy.config,
        velocity_model=policy.flow_matching.unet,
        output_root=cfg.output_dir,
    )

    # Roll through one episode
    max_episode_steps = env.call("spec")[0].max_episode_steps
    max_vis_steps = (max_episode_steps if cfg.vis.max_steps is None
                     else min(max_episode_steps, cfg.vis.max_steps))

    prog = trange(max_vis_steps, desc="Roll-out for visualisation")
    for step_idx in prog:
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

            # build global-conditioning with the policy’s helper
            global_cond = policy.flow_matching._prepare_global_conditioning(batch)

            vis.visualize(global_cond=global_cond)            

        observation, _, terminated, truncated, _ = env.step(action.cpu().numpy())

        # stop early if env terminates
        done |= (terminated | truncated)            # element-wise OR
        if done.all():                              # every env finished
            break

    logging.info(f"Finished in {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    init_logging()
    main()