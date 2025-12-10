import logging
import pickle
import random
from itertools import zip_longest
from pathlib import Path

import torch
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.fiper_rollout_scoring import FiperRolloutScoringPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.fiper_data_generator.fiper_rollout_scorer import FiperRolloutScorer
from lerobot.policies.factory import (
    make_flow_matching_adapter_from_policy,
    make_policy,
    make_pre_post_processors,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.uncertainty.uncertainty_scoring.scorer_artifacts import build_scorer_artifacts_for_fiper_scorer
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging


def interleave_paths(a: list[Path], b: list[Path]):
    for x, y in zip_longest(a, b):
        if x is not None:
            yield "calibration", x
        if y is not None:
            yield "test", y

def get_episode_idx(path: Path) -> int:
    """
    Extract the numeric episode index from filenames like:
    'episode_f_0146.pkl' or 'episode_s_0008.pkl'.
    """
    stem = path.stem
    idx_str = stem.split("_")[-1]
    return int(idx_str)

@parser.wrap()
def main(cfg: FiperRolloutScoringPipelineConfig):
    # Set global seed
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    # Random number generator to choose seeds for the single runs
    rng = random.Random(cfg.seed)

    allowed_policies = {"flow_matching", "smolvla"}
    if cfg.policy.type not in allowed_policies:
        raise ValueError(
            f"eval_uncertainty_estimation.py only supports policy types {allowed_policies}, "
            f"but got '{cfg.policy.type}'."
        )

    logging.info("Loading policy.")
    ds_meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
    )
    policy: PreTrainedPolicy = make_policy(
        cfg=cfg.policy,
        ds_meta=ds_meta,
    ).to(device)
    policy.eval()

    # Build preprocessing/postprocessing pipelines for observations/actions
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    scorer_artifacts = build_scorer_artifacts_for_fiper_scorer(
        fiper_data_scorer_cfg=cfg.fiper_rollout_scorer,
        policy=policy,
        preprocessor=preprocessor,
        dataset_cfg=cfg.dataset,
    )

    flow_matching_adapter = make_flow_matching_adapter_from_policy(policy)

    fiper_rollout_scorer = FiperRolloutScorer(
        config=cfg.fiper_rollout_scorer,
        flow_matching_adapter=flow_matching_adapter,
        scorer_artifacts=scorer_artifacts,
    )

    input_dir = Path(cfg.input_dir)
    input_rollout_dir = input_dir / "rollouts"
    input_calib_dir = input_rollout_dir / "calibration"
    input_test_dir = input_rollout_dir / "test"

    start_ep = cfg.start_episode
    end_ep = cfg.end_episode

    def in_range(p: Path) -> bool:
        idx = get_episode_idx(p)
        if start_ep is not None and idx < start_ep:
            return False
        if end_ep is not None and idx > end_ep:
            return False
        return True

    # Filter by episode range and then sort by episode id
    calib_files = [p for p in input_calib_dir.glob("*.pkl") if in_range(p)]
    test_files  = [p for p in input_test_dir.glob("*.pkl") if in_range(p)]

    calib_files.sort(key=get_episode_idx)
    test_files.sort(key=get_episode_idx)

    if not calib_files and not test_files:
        raise FileNotFoundError(
            f"No calibration or test files found in {input_calib_dir} or {input_test_dir}."
        )

    output_root = Path(cfg.output_dir)
    output_rollouts_dir = output_root / "rollouts"
    output_rollouts_dir.mkdir(parents=True, exist_ok=True)
    output_calib_dir = output_rollouts_dir / "calibration"
    output_test_dir = output_rollouts_dir / "test"
    output_calib_dir.mkdir(parents=True, exist_ok=True)
    output_test_dir.mkdir(parents=True, exist_ok=True)

    total_num_episodes = len(calib_files) + len(test_files)
    for split, pkl_path in tqdm(
        interleave_paths(calib_files, test_files),
        total=total_num_episodes,
        desc="Scoring rollouts",
        unit="file",
    ):
        logging.info(f"Scoring [{split}] {pkl_path.name}")

        with pkl_path.open("rb") as f:
            data = pickle.load(f)

        episode_metadata = data.get("metadata")
        rollout_data = data.get("rollout")

        # Reset state per episode/file
        fiper_rollout_scorer.reset()

        with torch.no_grad():
            fiper_rollout_scorer.score_rollout_data(rollout_data=rollout_data)

        # Save scored result into split-specific folder
        output_dir = output_calib_dir if split == "calibration" else output_test_dir
        fiper_rollout_scorer.save_data(output_dir=output_dir, episode_metadata=episode_metadata)

    logging.info("Done scoring all calibration/test rollouts.")

if __name__ == "__main__":
    init_logging()
    main()
