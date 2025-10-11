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
import logging
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from tqdm import trange

from lerobot.configs import parser
from lerobot.configs.visualize import VisualizePipelineConfig
from lerobot.constants import ACTION
from lerobot.envs.factory import make_single_env
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.factory import (
    make_flow_matching_adapter_from_policy,
    make_policy,
    make_pre_post_processors,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.io_utils import write_video
from lerobot.utils.live_window import LiveWindow
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging
from lerobot.visualizer import (
    ActionSeqVisualizer,
    FlowMatchingVisualizer,
    FlowVisualizer,
    NoiseToActionVisualizer,
    VectorFieldVisualizer,
)


def get_task_group_dir(
    out_root: Path,
    task_group: str,
):
    """
    Return the output directory for a given task group.
    """
    if "libero" in task_group:
        return out_root / task_group
    return out_root


def get_task_dir(
    out_root: Path,
    task_group: str,
    task_id: int,
) -> Path:
    """
    Return the output directory for a given task.
    """
    task_group_dir = get_task_group_dir(out_root, task_group)
    if "libero" in task_group:
        return task_group_dir / f"task{task_id:02d}"
    return task_group_dir


@parser.wrap()
def main(config: VisualizePipelineConfig):
    device = get_safe_torch_device(config.policy.device, log=True)

    # Set global seed
    if config.seed is not None:
        set_seed(config.seed)
        rollout_seeds = list(range(config.seed, config.seed + config.vis.num_rollouts))
    else:
        rollout_seeds = None

    allowed_policies = {"flow_matching", "smolvla"}
    if config.policy.type not in allowed_policies:
        raise ValueError(
            f"visualize_flow_matching.py only supports policy types {allowed_policies}, "
            f"but got '{config.policy.type}'."
        )

    logging.info("Creating environment")
    envs = make_single_env(config.env)

    policy: PreTrainedPolicy = make_policy(
        config.policy,
        env_cfg=config.env,
    ).to(device)
    policy.eval()

    # Build preprocessing/postprocessing pipelines for observations/actions
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config.policy,
        pretrained_path=config.policy.pretrained_path,
        # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    flow_matching_adapter = make_flow_matching_adapter_from_policy(policy=policy)

    num_rollouts = config.vis.num_rollouts
    progbar = trange(
        num_rollouts,
        desc=f"Visualizing for {num_rollouts} episodes."
    )
    for episode in progbar:
        # Flatten envs into list of (task_group, task_id, env)
        tasks = [(tg, tid, env) for tg, group in envs.items() for tid, env in group.items()]
        for task_group, task_id, env in tasks:
            generator = torch.Generator(device=device)
            if rollout_seeds is None:
                seed = None
            else:
                seed = rollout_seeds[episode]
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
            ep_dir = task_dir / f"rollout_{episode:03d}"

            # Prepare visualisers
            visualizers: list[FlowMatchingVisualizer] = []
            if "action_seq" in config.vis_types:
                visualizers.append(
                    ActionSeqVisualizer(
                        config=config.action_seq,
                        model=flow_matching_adapter,
                        postprocessor=postprocessor,
                        output_root=ep_dir,
                    )
                )
            if "flows" in config.vis_types:
                visualizers.append(
                    FlowVisualizer(
                        config=config.flows,
                        model=flow_matching_adapter,
                        output_root=ep_dir,
                    )
                )
            if "noise_to_action" in config.vis_types:
                visualizers.append(
                    NoiseToActionVisualizer(
                        config=config.noise_to_action,
                        model=flow_matching_adapter,
                        output_root=ep_dir,
                    )
                )
            if "vector_field" in config.vis_types:
                visualizers.append(
                    VectorFieldVisualizer(
                        config=config.vector_field,
                        model=flow_matching_adapter,
                        output_root=ep_dir,
                    )
                )

            # Roll through one episode
            if env.spec is None:
                max_episode_steps = env._max_episode_steps
            else:
                max_episode_steps = env.spec.max_episode_steps
            max_vis_steps = (
                max_episode_steps if config.vis.max_steps is None else min(max_episode_steps, config.vis.max_steps)
            )

            start_time = time.time()

            progbar = trange(
                max_vis_steps,
                desc=f"[Episode {episode+1}/{num_rollouts}]: Running rollout with at most {max_vis_steps} steps"
            )
            for step_idx in progbar:
                # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
                observation = preprocess_observation(observation)

                # Infer "task" from attributes of environments.
                observation = add_envs_task(env, observation)
                observation = preprocessor(observation)

                # Decide whether a new sequence will be generated
                new_action_gen = len(policy._queues[ACTION]) == 0

                with torch.no_grad():
                    action = policy.select_action(observation)
                action = postprocessor(action)

                if new_action_gen and (config.vis.start_step is None or step_idx >= config.vis.start_step):
                    for k in policy._queues:
                        if k != ACTION:
                            observation[k] = torch.stack(list(policy._queues[k]), dim=1)

                    for visualizer in visualizers:
                        visualizer.visualize(observation=observation, env=env, generator=generator)

                # Apply the next action
                observation, _, terminated, _, _ = env.step(action[0].cpu().numpy())
                render_frame(env)

                # Stop early if environment terminates
                if terminated:
                    break

            logging.info(f"Finished in {time.time() - start_time:.1f}s")

            env.close()

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
