import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.common import envs
from lerobot.common.policies.flow_matching.fiper_data_recording.configuration_fiper_data_recorder import (
    FiperDataRecorderConfig,
)
from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig


@dataclass
class FiperDataRecordingPipelineConfig:
    env: envs.EnvConfig
    fiper_data_recorder: FiperDataRecorderConfig = field(default_factory=FiperDataRecorderConfig)
    policy: PreTrainedConfig | None = None
    dataset: DatasetConfig | None = None

    n_calib_episodes: int = 60
    n_test_episodes: int = 240
    seed: int | None = None
    job_name: str | None = None
    output_dir: Path | None = None
    save_videos: bool = True

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            logging.warning(
                "No pretrained path was provided, policy for FIPER data recording will be built from "
                "scratch (random weights)."
            )

        if not self.job_name:
            if self.env is None:
                self.job_name = f"{self.policy.type}"
            else:
                self.job_name = f"{self.env.type}_{self.policy.type}"

        if not self.output_dir:
            now = dt.datetime.now()
            run_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/fiper_data_recording") / run_dir

        self.validate()

    def validate(self):
        pass

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]