import copy
import logging
from pathlib import Path
from typing import Any, List, Optional

import torch
from laplace import Laplace
from laplace.baselaplace import BaseLaplace
from torch.nn.utils import vector_to_parameters
from torch.utils.data import DataLoader

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.uncertainty.uncertainty_adapters.uncertainty_adapter import UncertaintyModelAdapter

from .laplace_wrappers.factory import make_laplace_wrapper
from .laplace_wrappers.laplace_wrapper import LaplaceWrapper


@torch.no_grad()
def create_laplace_calib_loader(
    laplace_wrapper: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    dataset_cfg: DatasetConfig,
    policy_cfg: PreTrainedConfig,
    calib_fraction: float,
    batch_size: int,
) -> DataLoader:
    """
    Build a DataLoader for fitting a Laplace approximation around the policy's underlying model.

    Args:
        laplace_wrapper: Wrapper around the pre-trained model compatible with laplace-torch.
        preprocessor: Preprocessor to apply to raw dataset samples.
        dataset_cfg: Dataset configuration.
        calib_fraction: Fraction of the full dataset to reserve for calibration (between 0 and 1).
        batch_size: Number of samples per batch in the returned DataLoader.
    """
    # Extract a subset of the full train dataset for calibration
    train_dataset = make_dataset(dataset_cfg=dataset_cfg, policy_cfg=policy_cfg)
    num_train_samples = len(train_dataset)
    num_calib_samples = int(calib_fraction * num_train_samples)
    calib_indices = torch.randperm(num_train_samples)[:num_calib_samples].tolist()
    calib_subset = torch.utils.data.Subset(train_dataset, calib_indices)

    calib_loader = torch.utils.data.DataLoader(
        calib_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=laplace_wrapper.build_collate_fn(preprocessor=preprocessor),
    )

    return calib_loader


def sample_adapter_from_posterior(
    laplace_posterior: BaseLaplace,
    uncertainty_adapter: UncertaintyModelAdapter,
    generator: Optional[torch.Generator] = None,
) -> UncertaintyModelAdapter:
    """
    Draw one weight sample from a fitted Laplace posterior and return a cloned
    uncertainty adapter whose underlying model uses those sampled weights.

    Args:
        laplace_posterior: Fitted Laplace posterior over the model's trainable params.
        uncertainty_adapter: Source adapter whose model defines architecture
            and parameter ordering for writing sampled weights.
        generator: Optional RNG for reproducibility.

    Returns:
        An adapter with its model's trainable parameters replaced by one Monte Carlo sample from the Laplace posterior.
    """
    # Draw weights from the Laplace posterior
    laplace_model_weights = laplace_posterior.sample(
        n_samples=1,
        generator=generator
    ).squeeze(0)

    # Copy the MAP model so we never mutate the original
    laplace_adapter = copy.deepcopy(uncertainty_adapter)

    # Collect the parameters that were in the posterior
    target_params = [p for p in laplace_adapter.model.parameters() if p.requires_grad]

    # Consistency check to avoid silent weight mis-alignment
    n_expected = sum(p.numel() for p in target_params)
    if laplace_model_weights.numel() != n_expected:
        raise RuntimeError(
            f"[Laplace] Sample size mismatch: drew {laplace_model_weights.numel()} "
            f"weights but found {n_expected} trainable parameters in the copy."
        )

    # Write sampled parameters into the copied model (in-place assignment)
    vector_to_parameters(laplace_model_weights, target_params)

    # Move the model to the same device as sampled weights and switch to inference mode
    laplace_adapter.model = laplace_adapter.model.to(laplace_model_weights.device)
    laplace_adapter.model.eval()

    return laplace_adapter


def make_laplace_path(
    laplace_wrapper: LaplaceWrapper,
    pretrained_path: Path | str,
    calib_fraction: float,
) -> Path:
    """
    Build (and create) the on-disk path where we save/load a Laplace posterior.
    """
    scope_abbreviations = sorted([laplace_wrapper.scope_abbr[scope] for scope in laplace_wrapper.scopes])
    calib_fraction_pct = int(calib_fraction * 100)
    filename = f"laplace_{'_'.join(scope_abbreviations)}_frac{calib_fraction_pct}pct.bin"
    return Path(pretrained_path) / filename

def get_laplace_posterior(
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    laplace_scopes: List[str],
    calib_fraction: float,
    batch_size: int,
    dataset_cfg: DatasetConfig,
) -> Laplace:
    """
    Construct or load a Laplace posterior for the underlying model of a policy.

    Builds a calibration DataLoader if needed and fits a new posterior,
    otherwise loads an existing one from disk.

    Returns:
        A fitted or loaded Laplace posterior.
    """
    # Wrap the flow matching model so it takes inputs and generates outputs compatible with Laplace
    laplace_wrapper = make_laplace_wrapper(policy=policy, scopes=laplace_scopes)

    laplace_path = make_laplace_path(
        laplace_wrapper=laplace_wrapper,
        pretrained_path=policy.config.pretrained_path,
        calib_fraction=calib_fraction,
    )

    laplace_posterior = Laplace(
        laplace_wrapper,
        likelihood="regression",
        subset_of_weights="all",  # uses only params with requires_grad=True
        hessian_structure="diag",
    )

    if laplace_path.exists():
        logging.info(f"Loading Laplace posterior from {laplace_path}")
        laplace_posterior.load_state_dict(torch.load(laplace_path))
    else:
        logging.info("Create Laplace calibration loader.")
        calib_loader = create_laplace_calib_loader(
            laplace_wrapper=laplace_wrapper,
            preprocessor=preprocessor,
            dataset_cfg=dataset_cfg,
            policy_cfg=policy.config,
            calib_fraction=calib_fraction,
            batch_size=batch_size,
        )

        logging.info("Fitting new Laplace posterior.")
        laplace_posterior.fit(calib_loader)

        logging.info(f"Save Laplace posterior to {laplace_path}")
        torch.save(laplace_posterior.state_dict(), laplace_path)

    return laplace_posterior
