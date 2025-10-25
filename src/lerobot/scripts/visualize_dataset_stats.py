"""
python -m lerobot.scripts.visualize_dataset_stats \
  --repo-id HuggingFaceVLA/libero \
  --libero-tasks '{"libero_spatial":[0,1,2,3,4,5,6,7,8,9],"libero_object":[0,1,2,3,4,5,6,7,8,9],"libero_goal":[0,1,2,3,4,5,6,7,8,9],"libero_10":[1,2,3,5,6]}' \
  --features actions \
  --output-dir outputs/visualize_dataset_stats
"""


import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

from pathlib import Path
from typing import Dict, Any, Sequence

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import filter_libero_episodes


def compute_feature_histograms_for_episodes(
    dataset, 
    episode_ids: Sequence[int],
    *,
    features: Dict[str, Dict[str, Any]] | None = None,
    bins: int = 100,
    eps: float = 1e-8,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Aggregate per-dimension histograms for numeric features (e.g., states/actions) over a subset of episodes.

    Args:
        dataset: A LeRobotDataset instance.
        episode_ids: Episode indices to include.
        features: Optional subset of dataset.meta.features to histogram. If None,
                  all non-string/non-image/non-video features are used.
        bins: Number of histogram bins.
        eps: Small padding added to [min, max] when the range collapses (constant-valued dims).

    Returns:
        A dict keyed by feature name. Each value is a dict with:
          - "counts":  float64 array of shape (D, bins) with aggregated counts per dimension.
          - "bin_edges": float64 array of shape (D, bins+1) with the bin edges per dimension.
          - "dim_names": list of length D with human-readable names if available (fallbacks to dim_0, ...).
    """
    # 1) Pick numeric features only by default
    if features is None:
        features = {
            k: spec for k, spec in dataset.meta.features.items()
            if spec.get("dtype") not in {"string", "image", "video"}
        }

    hf = dataset.hf_dataset
    ep_from = dataset.meta.episodes["dataset_from_index"]
    ep_to   = dataset.meta.episodes["dataset_to_index"]

    # 2) First pass: compute min/max for each feature dim across selected episodes
    fmins, fmaxs = {}, {}
    fdims = {}
    for eid in episode_ids:
        start, end = int(ep_from[eid]), int(ep_to[eid])
        rows = hf[start:end]
        for key in features:
            vals = rows[key]
            if len(vals) == 0:
                continue
            # Stack to (T, D) or (T,) -> then reshape to (T, D)
            if isinstance(vals[0], torch.Tensor):
                arr = torch.stack(vals).cpu().numpy()
            else:
                arr = np.asarray(vals)
            if arr.ndim == 1:
                arr = arr[:, None]
            T, D = arr.shape[0], arr.shape[1]
            fdims[key] = D
            cur_min = np.nanmin(arr, axis=0)
            cur_max = np.nanmax(arr, axis=0)
            if key not in fmins:
                fmins[key] = cur_min
                fmaxs[key] = cur_max
            else:
                fmins[key] = np.minimum(fmins[key], cur_min)
                fmaxs[key] = np.maximum(fmaxs[key], cur_max)

    # 3) Prepare fixed bin edges per feature dim so we can aggregate counts
    result = {}
    for key, D in fdims.items():
        mins = fmins[key].astype(np.float64)
        maxs = fmaxs[key].astype(np.float64)
        # Guard constant dims
        same = (maxs - mins) <= 0
        mins[same] -= eps
        maxs[same] += eps

        edges = np.empty((D, bins + 1), dtype=np.float64)
        for d in range(D):
            edges[d] = np.linspace(mins[d], maxs[d], bins + 1)
        result[key] = {
            "counts": np.zeros((D, bins), dtype=np.float64),
            "bin_edges": edges,
            "dim_names": _safe_dim_names(dataset, key, D),
        }

    # 4) Second pass: accumulate hist counts per episode using fixed edges
    for eid in episode_ids:
        start, end = int(ep_from[eid]), int(ep_to[eid])
        rows = hf[start:end]
        for key, out in result.items():
            vals = rows[key]
            if len(vals) == 0:
                continue
            if isinstance(vals[0], torch.Tensor):
                arr = torch.stack(vals).cpu().numpy()
            else:
                arr = np.asarray(vals)
            if arr.ndim == 1:
                arr = arr[:, None]
            # Iterate per-dimension with fixed edges
            for d in range(arr.shape[1]):
                counts, _ = np.histogram(arr[:, d], bins=out["bin_edges"][d])
                out["counts"][d] += counts

    return result


def _safe_dim_names(dataset, key: str, D: int):
    # Try to use names from metadata if present; otherwise fallback
    names = dataset.meta.names.get(key)
    if isinstance(names, list) and len(names) == D:
        return names
    if isinstance(names, dict):  # sometimes names are dicts; try to flatten sensible order
        try:
            ordered = [names[i] for i in range(D)]
            return ordered
        except Exception:
            pass
    return [f"dim_{i}" for i in range(D)]


def plot_feature_histograms(histograms, keys=None, max_dims=8):
    """
    Plot aggregated histograms returned by compute_feature_histograms_for_episodes.
    """
    keys = keys or list(histograms.keys())
    for key in keys:
        data = histograms[key]
        counts, edges, names = data["counts"], data["bin_edges"], data["dim_names"]
        D = counts.shape[0]
        n = min(D, max_dims)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 3.2*rows))
        axes = np.atleast_1d(axes).ravel()
        for i in range(n):
            ax = axes[i]
            centers = 0.5 * (edges[i, 1:] + edges[i, :-1])
            ax.bar(centers, counts[i], width=np.diff(edges[i]), align="center")
            ax.set_title(f"{key} • {names[i]}")
            ax.set_xlabel("value"); ax.set_ylabel("count")
        for j in range(n, len(axes)):
            axes[j].axis("off")
        fig.tight_layout()
        plt.show()


# ----------------------------
# CLI
# ----------------------------
def _parse_episode_ids(spec: str, total_episodes: int) -> list[int]:
    """
    Parse episode id spec:
      - "1,2,3" -> [1,2,3]
      - "5:10"  -> [5,6,7,8,9]
      - "10:"   -> [10, ..., total_episodes-1]
      - ":20"   -> [0, ..., 19]
      - ":"     -> all episodes
    """
    spec = spec.strip()
    if "," in spec:
        return [int(s) for s in spec.split(",") if s != ""]
    if ":" in spec:
        left, right = spec.split(":", 1)
        start = int(left) if left != "" else 0
        stop  = int(right) if right != "" else total_episodes
        return list(range(start, min(stop, total_episodes)))
    # single int
    return [int(spec)]


def main():
    parser = argparse.ArgumentParser(description="Histogram visualizer for LeRobotDataset episode subsets.")
    parser.add_argument("--repo-id", type=str, required=True, help="Name of hugging face repository containing a LeRobotDataset dataset (e.g. `lerobot/pusht`).")
    parser.add_argument("--root", type=str, default=None, help="Root directory for the dataset stored locally (e.g. `--root data`). By default, the dataset will be loaded from hugging face cache folder, or downloaded from the hub if available.")
    parser.add_argument("--libero_tasks", type=str, default=None, help="Optionally choose LIBERO tasks to filter dataset episodes.")
    parser.add_argument("--features", type=list, default=None, help="Feature keys to include.")
    parser.add_argument("--hbins", type=int, default=100, help="Number of histogram bins.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory path to save a picture of the histogram.")

    args = parser.parse_args()

    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.root,
    )

    features = None
    feature_keys = None
    if args.features:
        features = {
            k: dataset.meta.features[k]
            for k in args.features
            if k in dataset.meta.features and dataset.meta.features[k].get("dtype") not in {"string", "image", "video"}
        }
        feature_keys = list(features.keys())

    all_episode_ids = list(dataset.meta.episodes["episode_index"])
    if args.repo_id == "HuggingFaceVLA/libero" and args.libero_tasks is not None:
        episode_ids_to_use = filter_libero_episodes(
            dataset=dataset,
            tasks_to_use=args.libero_tasks,
        )
    else:
        episode_ids_to_use = all_episode_ids

    hists = compute_feature_histograms_for_episodes(
        dataset=dataset,
        episode_ids=episode_ids_to_use,
        features=features,
        bins=args.hbins,
    )

    # Optionally save raw histogram arrays
    if args.save_dir:
        outdir = Path(args.save_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        for key, data in hists.items():
            np.savez(
                outdir / f"{key}_hist.npz",
                counts=data["counts"],
                bin_edges=data["bin_edges"],
                dim_names=np.array(data["dim_names"], dtype=object),
            )
        print(f"Saved hist arrays to: {outdir}")

    # Plot (unless suppressed)
    if not args.no_show:
        plot_feature_histograms(hists, keys=keys_for_plot, max_dims=args.max_dims)


if __name__ == "__main__":
    main()
