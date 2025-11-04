#!/usr/bin/env python
"""
Visualize the cross Bayesian uncertainty estimation method by creating a vector field
plot of the scorer model overlaid by the actions from the sampler model.

Usage example:

Create the ensembling uncertainty estimation visualization for the Push-T task.

```
python src/lerobot/scripts/uncertainty/visualize_bayesian_sampler.py \
    --policy.path=outputs/train/flow_matching/pusht/checkpoints/last/pretrained_model \
    --policy.device=cuda \
    --env.type=pusht
```
"""
import logging
import time
from pathlib import Path
from typing import Dict

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm, trange

from lerobot.configs import parser
from lerobot.configs.visualize_bayesian_sampler import VisualizeBayesianSamplerPipelineConfig
from lerobot.constants import ACTION
from lerobot.envs.factory import make_single_env
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.common.flow_matching.adapter import BaseFlowMatchingAdapter
from lerobot.policies.common.flow_matching.ode_solver import ODESolver
from lerobot.policies.factory import (
    make_flow_matching_adapter_from_policy,
    make_policy,
    make_pre_post_processors,
)
from lerobot.uncertainty.uncertainty_samplers.configuration_uncertainty_sampler import (
    UncertaintySamplerConfig,
)
from lerobot.uncertainty.uncertainty_samplers.uncertainty_sampler import UncertaintySampler
from lerobot.uncertainty.uncertainty_scoring.scorer_artifacts import (
    build_scorer_artifacts_for_uncertainty_sampler,
)
from lerobot.utils.io_utils import get_task_dir, write_video
from lerobot.utils.live_window import LiveWindow
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging
from lerobot.visualizers import (
    ActionSeqVisualizer,
    FlowVisualizer,
    VectorFieldVisualizer,
)


@parser.wrap()
def main(config: VisualizeBayesianSamplerPipelineConfig):
    device = get_safe_torch_device(config.policy.device, log=True)

    # Set global seed
    if config.seed is not None:
        set_seed(config.seed)
        rollout_seeds = list(range(config.seed, config.seed + config.vis.num_rollouts))
    else:
        rollout_seeds = None

    # Plug in the cross-bayesian sampler config into the uncertainty sampler config
    uncertainty_sampler_config = UncertaintySamplerConfig()
    uncertainty_sampler_config.type = "cross_bayesian"
    uncertainty_sampler_config.cross_bayesian_sampler = config.cross_bayesian_sampler

    allowed_policies = {"flow_matching", "smolvla"}
    if config.policy.type not in allowed_policies:
        raise ValueError(
            f"visualize_bayesian_sampler.py only supports policy types {allowed_policies}, "
            f"but got '{config.policy.type}'."
        )

    logging.info("Creating environment")
    envs = make_single_env(config.env)

    logging.info("Creating policy")
    policy = make_policy(
        config.policy,
        env_cfg=config.env,
    ).to(device)
    policy.eval()

    sampler_model = make_flow_matching_adapter_from_policy(policy=policy)

    # Build preprocessing/postprocessing pipelines for observations/actions
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config.policy,
        pretrained_path=config.policy.pretrained_path,
        # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    # Prepare scorer artifacts for cross-bayesian uncertainty sampler
    scorer_artifacts = build_scorer_artifacts_for_uncertainty_sampler(
        uncertainty_sampler_cfg=uncertainty_sampler_config,
        policy=policy,
        preprocessor=preprocessor,
        dataset_cfg=config.dataset,
    )

    # Initialize and extract the cross-bayesian sampler
    policy.init_uncertainty_sampler(
        config=uncertainty_sampler_config,
        scorer_artifacts=scorer_artifacts
    )
    cross_bayesian_sampler: UncertaintySampler = policy.uncertainty_sampler

    num_rollouts = config.vis.num_rollouts
    episode_progbar = trange(
        num_rollouts,
        desc=f"Visualizing for {num_rollouts} episodes."
    )
    for ep in episode_progbar:
        # Flatten envs into list of (task_group, task_id, env)
        tasks = [(tg, tid, env) for tg, group in envs.items() for tid, env in group.items()]
        for task_group, task_id, env in tasks:
            generator = torch.Generator(device=device)
            if rollout_seeds is None:
                seed = None
            else:
                seed = rollout_seeds[ep]
                generator.manual_seed(seed)

            reset_kwargs: dict = {}
            if config.start_state is not None and config.env.type == "pusht":
                logging.info(f"Resetting to provided start_state {config.start_state}")
                reset_kwargs["options"] = {"reset_to_state": config.start_state}

            # Reset the policy and environment
            policy.reset()
            observation, _ = env.reset(seed=seed)

            # Cache frames for creating video
            video_frames: list[np.ndarray] = []
            # Setup for live visualization
            if config.show:
                live_view = LiveWindow("Live Visualization")

            # Callback for visualization.
            def render_frame(env: gym.Env) -> np.ndarray:
                rgb_frame = env.render()
                video_frames.append(rgb_frame)

                # Live visualization
                if config.show:
                    live_view.enqueue_frame(rgb_frame[..., ::-1])

                return rgb_frame

            render_frame(env)

            task_dir = get_task_dir(
                out_root=Path(config.output_dir),
                task_group=task_group,
                task_id=task_id,
            )
            ep_dir = task_dir / f"rollout_{ep:03d}"

            # Prepare visualizers
            sampler_action_seq_visualizer = ActionSeqVisualizer(
                config=config.action_seq,
                model=sampler_model,
                postprocessor=postprocessor,
                output_root=ep_dir,
            )
            scorer_action_seq_visualizer = ActionSeqVisualizer(
                config.action_seq,
                model=None,
                postprocessor=postprocessor,
                output_root=ep_dir,
            )
            vector_field_visualizer = VectorFieldVisualizer(
                config=config.vector_field,
                model=None,
                output_root=ep_dir,
            )
            flow_visualizer = FlowVisualizer(
                config=config.flows,
                model=sampler_model,
                output_root=ep_dir,
            )

            # Initialize the dictionary of actions to visualize in the vector field plot
            action_data: Dict[str, Tensor] = {}

            # Roll through one episode
            if env.spec is None:
                max_episode_steps = env._max_episode_steps
            else:
                max_episode_steps = env.spec.max_episode_steps
            max_vis_steps = (
                max_episode_steps if config.vis.max_steps is None else min(max_episode_steps, config.vis.max_steps)
            )

            start_time = time.time()

            action_generation_iter = 0
            step_progbar = trange(
                max_vis_steps,
                desc=f"[Episode {ep+1}/{num_rollouts}]: Running rollout with at most {max_vis_steps} steps"
            )
            for step_idx in step_progbar:
                # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
                observation = preprocess_observation(observation)

                # Infer "task" from attributes of environments.
                observation = add_envs_task(env, observation)
                observation = preprocessor(observation)

                # Decide whether a new sequence will be generated
                new_action_gen = len(policy._queues["action"]) == 0

                if new_action_gen and (config.vis.start_step is None or step_idx >= config.vis.start_step):
                    action_generation_iter += 1
                    tqdm.write(f"-----------------------Action Generation Iteration {action_generation_iter}---------------------")

                with torch.no_grad():
                    action = policy.select_action(observation)
                action = postprocessor(action)

                if new_action_gen and (config.vis.start_step is None or step_idx >= config.vis.start_step):
                    # Clear action data dictionary
                    action_data.clear()

                    for k in policy._queues:
                        if k != ACTION:
                            observation[k] = torch.stack(list(policy._queues[k]), dim=1)

                    # Sample actions and get their uncertainties based on the scorer model
                    _, uncertainty = cross_bayesian_sampler.conditional_sample_with_uncertainty(
                        observation=observation, generator=generator
                    )

                    # Extract the current scorer model
                    scorer_model: BaseFlowMatchingAdapter = cross_bayesian_sampler.scorer_models[0]

                    # Visualize action sequence batch of sampler and scorer model
                    sampler_action_seq_visualizer.visualize(
                        observation=observation, env=env, dir_name="sampler_action_seq", generator=generator
                    )
                    scorer_action_seq_visualizer.model = scorer_model
                    scorer_action_seq_visualizer.visualize(
                        observation=observation, env=env, dir_name="scorer_action_seq", generator=generator
                    )

                    # Build the velocity functions for sampler and scorer conditioned on the current observation
                    num_samples = config.cross_bayesian_sampler.num_action_samples
                    sampler_conditioning = sampler_model.prepare_conditioning(observation, num_samples)
                    sampler_velocity_fn = sampler_model.make_velocity_fn(conditioning=sampler_conditioning)

                    scorer_conditioning = scorer_model.prepare_conditioning(observation, num_samples)
                    scorer_velocity_fn = scorer_model.make_velocity_fn(conditioning=scorer_conditioning)

                    if config.cross_bayesian_sampler.scoring_metric == "inter_vel_diff":
                        flow_visualizer.visualize_velocity_difference(
                            sampler_velocity_fn=sampler_velocity_fn,
                            scorer_velocity_fn=scorer_velocity_fn,
                            velocity_eval_times=torch.tensor(
                                config.cross_bayesian_sampler.velocity_eval_times,
                                device=device
                            )
                        )
                    else:
                        # Sample actions with the scorer model to compare with the sampler actions
                        noise_sample = scorer_model.sample_prior(
                            num_samples=num_samples,
                            generator=generator,
                        )
                        ode_solver = ODESolver()
                        scorer_actions = ode_solver.sample(
                            x_0=noise_sample,
                            velocity_fn=scorer_velocity_fn,
                            method=scorer_model.ode_solver_config["solver_method"],
                            step_size=scorer_model.ode_solver_config["step_size"],
                            atol=scorer_model.ode_solver_config["atol"],
                            rtol=scorer_model.ode_solver_config["rtol"],
                        )

                        # Store the action samples to overlay them in the vector field plot
                        action_data["Scorer Actions"] = scorer_actions
                        action_data["Sampler Actions"] = cross_bayesian_sampler.action_candidates[1:]

                        # Choose an action which will be used to generate the vector field plot
                        action_data["Base Action"] = cross_bayesian_sampler.action_candidates[0].unsqueeze(0)

                        # Visualize scorer vector field with sampler action sequences
                        vector_field_visualizer.model = scorer_model
                        vector_field_visualizer.visualize(
                            observation=observation,
                            visualize_actions=True,
                            actions=action_data,
                            uncertainty=uncertainty,
                            generator=generator
                        )

                # Apply the next action
                observation, _, terminated, _, _ = env.step(action[0].cpu().numpy())
                render_frame(env)

                # Stop early if environment terminates
                if terminated:
                    break

        logging.info(f"Finished in {time.time() - start_time:.1f}s")

        # Close the live visualization
        if config.show:
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
