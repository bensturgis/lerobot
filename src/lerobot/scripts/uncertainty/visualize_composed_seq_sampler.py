#!/usr/bin/env python
"""
Visualize the composed action sequence uncertainty estimation method by creating a vector field plot overlaid by
the composed action sequences and noise-to-action transformations for consecutive action generation steps.

Usage example:

Create the composed action sequence visualization for the Push-T task.

```
python src/lerobot/scripts/uncertainty/visualize_composed_seq_sampler.py \
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
import matplotlib.cm as cm
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm, trange

from lerobot.configs import parser
from lerobot.configs.visualize_composed_seq_sampler import VisualizeComposedSeqSamplerPipelineConfig
from lerobot.constants import ACTION
from lerobot.envs.factory import make_single_env
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.common.flow_matching.ode_solver import select_ode_states
from lerobot.policies.factory import (
    make_flow_matching_adapter_from_policy,
    make_policy,
    make_pre_post_processors,
)
from lerobot.uncertainty.uncertainty_samplers.composed_seq_sampler import ComposedSequenceSampler
from lerobot.uncertainty.uncertainty_samplers.configuration_uncertainty_sampler import (
    UncertaintySamplerConfig,
)
from lerobot.uncertainty.uncertainty_samplers.utils import compose_ode_states
from lerobot.uncertainty.uncertainty_scoring.scorer_artifacts import (
    build_scorer_artifacts_for_uncertainty_sampler,
)
from lerobot.utils.io_utils import get_task_dir, write_video
from lerobot.utils.live_window import LiveWindow
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging
from lerobot.visualizers import (
    ActionSeqVisualizer,
    NoiseToActionVisualizer,
    VectorFieldVisualizer,
)


@parser.wrap()
def main(config: VisualizeComposedSeqSamplerPipelineConfig):
    device = get_safe_torch_device(config.policy.device, log=True)

    # Set global seed
    if config.seed is not None:
        set_seed(config.seed)
        rollout_seeds = list(range(config.seed, config.seed + config.vis.num_rollouts))
    else:
        rollout_seeds = None

    # Plug in the composed sequence sampler config into the uncertainty sampler config
    uncertainty_sampler_config = UncertaintySamplerConfig()
    uncertainty_sampler_config.type = "composed_sequence"
    uncertainty_sampler_config.composed_sequence_sampler = config.composed_sequence_sampler

    allowed_policies = {"flow_matching", "smolvla"}
    if config.policy.type not in allowed_policies:
        raise ValueError(
            f"visualize_composed_seq_sampler.py only supports policy types {allowed_policies}, "
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

    flow_matching_adapter = make_flow_matching_adapter_from_policy(policy=policy)

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

    # Initialize composed action sequence uncertainty sampler
    policy.init_uncertainty_sampler(
        config=uncertainty_sampler_config,
        scorer_artifacts=scorer_artifacts
    )
    composed_seq_sampler: ComposedSequenceSampler = policy.uncertainty_sampler

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
            action_seq_visualizer = ActionSeqVisualizer(
                config=config.action_seq,
                model=flow_matching_adapter,
                postprocessor=postprocessor,
                output_root=ep_dir,
            )
            noise_to_action_visualizer = NoiseToActionVisualizer(
                config=config.noise_to_action,
                model=flow_matching_adapter,
                output_root=ep_dir,
            )
            vector_field_visualizer = VectorFieldVisualizer(
                config=config.vector_field,
                model=flow_matching_adapter,
                output_root=ep_dir,
            )

            # Only visualize the attached action steps of the composed action sequence
            prev_action_seq_end = (
                flow_matching_adapter.n_obs_steps - 1 + flow_matching_adapter.n_action_steps
            )
            horizon = flow_matching_adapter.horizon
            attached_action_steps = list(range(prev_action_seq_end, horizon))
            vector_field_visualizer.action_steps = [
                step for step in vector_field_visualizer.action_steps
                if step in attached_action_steps
            ]
            if len(vector_field_visualizer.action_steps) == 0:
                vector_field_visualizer.action_steps = list(range(prev_action_seq_end, horizon))

            # Initialize the dictionary of actions to visualize in the vector field plot
            action_data: Dict[str, Tensor] = {}

            # At the beginning of an epsiode we don't have previous ODE states to compose with
            prev_ode_states: Tensor | None = None
            prev_selected_action_idx: int | None = None

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

                    # Get the ODE states, actions and uncertainties from the current action generation iteration
                    new_ode_states = composed_seq_sampler.prev_ode_states
                    new_selected_action_idx = composed_seq_sampler.prev_selected_action_idx
                    uncertainty = composed_seq_sampler.uncertainty

                    # Compose actions
                    if prev_ode_states is not None:
                        composed_ode_states = compose_ode_states(
                            prev_ode_states=prev_ode_states[
                                :, prev_selected_action_idx:prev_selected_action_idx+1, :, :
                            ],
                            new_ode_states=new_ode_states,
                            horizon=flow_matching_adapter.horizon,
                            n_action_steps=flow_matching_adapter.n_action_steps,
                            n_obs_steps=flow_matching_adapter.n_obs_steps,
                        )

                        # Choose an action which will be used to generate the vector field plot
                        action_data["Base Action"] = composed_ode_states[-1, 0].unsqueeze(0)
                        # Store action samples, composed action and the selected action sample for visualization
                        action_data["Action Samples"] = torch.cat((
                            prev_ode_states[-1, :prev_selected_action_idx],
                            prev_ode_states[-1, prev_selected_action_idx+1:]
                        ))
                        action_data["Composed Actions"] = composed_ode_states[-1, 1:]
                        action_data["Selected Action Sample"] = prev_ode_states[-1, prev_selected_action_idx].unsqueeze(0)

                        # Visualize vector field with composed action sequences
                        vector_field_visualizer.visualize(
                            observation=observation,
                            visualize_actions=True,
                            actions=action_data,
                            uncertainty=uncertainty,
                            generator=generator
                        )

                        # Setup to visualize noise-to-action transformation
                        noise_to_action_visualizer._update_run_dir()
                        combined_horizon = horizon + flow_matching_adapter.n_action_steps
                        cmap = cm.get_cmap('plasma')
                        colors = cmap(torch.arange(combined_horizon) / (combined_horizon - 1))
                        step_labels = ("t", *[f"t+{k}" for k in range(1, combined_horizon)])

                        # Extract the ODE states fitting the visualizer's evaluation times
                        selected_prev_ode_states, noise_to_action_visualizer.ode_eval_times = select_ode_states(
                            time_grid=composed_seq_sampler.sampling_time_grid,
                            ode_states=prev_ode_states,
                            requested_times=noise_to_action_visualizer.ode_eval_times,
                        )

                        selected_new_ode_states, noise_to_action_visualizer.ode_eval_times = select_ode_states(
                            time_grid=composed_seq_sampler.sampling_time_grid,
                            ode_states=new_ode_states,
                            requested_times=noise_to_action_visualizer.ode_eval_times,
                        )

                        prev_actions_overlay = {
                            "label": "Previous ODE States",
                            "ode_states": selected_prev_ode_states.transpose(0, 1)[
                                prev_selected_action_idx:prev_selected_action_idx+1, :, :, :
                            ],
                            "colors": colors[:horizon],
                            # "step_labels": step_labels[:self.horizon],
                            "text_kwargs": {"xytext": (-14, -12)},
                            "scale": 60,
                        }
                        new_actions_overlay = {
                            "label": "Current ODE States",
                            "ode_states": selected_new_ode_states.transpose(0, 1)[
                                new_selected_action_idx:new_selected_action_idx+1, :, :, :
                            ],
                            "colors": colors[(horizon - flow_matching_adapter.n_action_steps):],
                            # "step_labels": step_labels[new_noise_overlap_end:],
                            "text_kwargs": {"xytext": (2, 2)},
                            "scale": 80,
                            "marker": "x",
                        }
                        cbar_kwargs = {
                            "cmap": cmap,
                            "horizon": combined_horizon,
                        }
                        noise_to_action_visualizer.plot_noise_to_action_overlays(
                            action_overlays=[prev_actions_overlay, new_actions_overlay],
                            uncertainty=uncertainty,
                            cbar_kwargs=cbar_kwargs,
                        )
                        noise_to_action_visualizer._create_gif()

                    # Visualize action sequence batch
                    action_seq_visualizer.visualize(observation=observation, env=env, generator=generator)

                    # Set the previous global conditioning vector and the previous action sequences to compose with
                    prev_ode_states = new_ode_states
                    prev_selected_action_idx = new_selected_action_idx

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
