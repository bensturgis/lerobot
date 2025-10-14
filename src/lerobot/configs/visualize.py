import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path

from lerobot import envs
from lerobot.configs import parser
from lerobot.configs.default import VisConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.visualizers.configuration_visualizer import (
    ActionSeqVisConfig,
    FlowVisConfig,
    NoiseToActionVisConfig,
    VectorFieldVisConfig,
)


@dataclass
class VisualizePipelineConfig:
    env: envs.EnvConfig
    policy: PreTrainedConfig | None = None
    vis: VisConfig = field(default_factory=VisConfig)
    action_seq: ActionSeqVisConfig = field(default_factory=ActionSeqVisConfig)
    flows: FlowVisConfig = field(default_factory=FlowVisConfig)
    noise_to_action: NoiseToActionVisConfig = field(default_factory=NoiseToActionVisConfig)
    vector_field: VectorFieldVisConfig = field(default_factory=VectorFieldVisConfig)

    # Which visualizers to run (you can pick one or more)
    vis_types: list[str] = field(
        default_factory=lambda: ["action_seq", "flows", "vector_field"]
    )

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

        if "flows" in self.vis_types and self.flows.action_dim_names is None:
            self.flows.action_dim_names = self.vis.action_dim_names

        if "noise_to_action" in self.vis_types and self.noise_to_action.action_dim_names is None:
            self.noise_to_action.action_dim_names = self.vis.action_dim_names

        if "vector_field" in self.vis_types and self.vector_field.action_dim_names is None:
            self.vector_field.action_dim_names = self.vis.action_dim_names

        if not self.job_name:
            if self.env is None:
                self.job_name = f"{self.policy.type}"
            else:
                self.job_name = f"{self.env.type}_{self.policy.type}"

        if not self.output_dir:
            now = dt.datetime.now()
            vis_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/flow_matching_visualizations") / vis_dir

        self.validate()

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]

    def validate(self):
        # vis_types check
        allowed_vis = {"flows", "vector_field", "action_seq", "noise_to_action"}
        for v in self.vis_types:
            if v not in allowed_vis:
                raise ValueError(
                    f"Unknown visualization type '{v}'. "
                    f"Allowed: {sorted(allowed_vis)}"
                )

        active_action_dims = {
            vis_type: tuple(getattr(self, vis_type).action_dims)
            for vis_type in ("flows", "vector_field", "noise_to_action")
            if vis_type in self.vis_types
        }

        if len(set(active_action_dims.values())) > 1:
            logging.warning(
                f"Visualizing different action dimensions across visualizers: {active_action_dims}"
            )
