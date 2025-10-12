import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path

from lerobot import envs
from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig, VisConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.uncertainty.uncertainty_samplers.configuration_uncertainty_sampler import (
    ComposedSequenceSamplerConfig,
)
from lerobot.visualizer.configuration_visualizer import (
    ActionSeqVisConfig,
    NoiseToActionVisConfig,
    VectorFieldVisConfig,
)


@dataclass
class VisualizeComposedSeqSamplerPipelineConfig:
    env: envs.EnvConfig
    policy: PreTrainedConfig | None = None
    composed_sequence_sampler: ComposedSequenceSamplerConfig = field(default_factory=ComposedSequenceSamplerConfig)
    dataset: DatasetConfig | None = None
    vis: VisConfig = field(default_factory=VisConfig)
    action_seq: ActionSeqVisConfig = field(default_factory=ActionSeqVisConfig)
    vector_field: VectorFieldVisConfig = field(default_factory=VectorFieldVisConfig)
    noise_to_action: NoiseToActionVisConfig = field(default_factory=NoiseToActionVisConfig)

    seed: int | None = None
    job_name: str | None = None
    output_dir: Path | None = None

    # `show` enables live visualization of the first environment during evaluation
    show: bool = False

    # Optional custom start state for PushT-v0: 
    # [agent_x, agent_y, block_x, block_y, block_theta]
    start_state: list[float] | None = None

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            logging.warning(
                "No pretrained path was provided, visualized policy will be built from scratch (random weights)."
            )

        if self.vector_field.action_dim_names is None:
            self.vector_field.action_dim_names = self.vis.action_dim_names

        if self.noise_to_action.action_dim_names is None:
            self.noise_to_action.action_dim_names = self.vis.action_dim_names

        # Plot the velocity evaluation times of the composed sequence sampler in the noise-to-action visualization
        self.noise_to_action.ode_eval_times = list(self.composed_sequence_sampler.scoring_metric.velocity_eval_times)
        if self.noise_to_action.ode_eval_times[-1] != 1.0:
            self.noise_to_action.ode_eval_times.append(1.0)

        if not self.job_name:
            if self.env is None:
                self.job_name = f"{self.policy.type}"
            else:
                self.job_name = f"{self.env.type}_{self.policy.type}"

        if not self.output_dir:
            now = dt.datetime.now()
            vis_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/composed_seq_sampler_visualizations") / vis_dir

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]