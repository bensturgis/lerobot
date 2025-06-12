import numpy as np
import random
import warnings
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym


class PerturbationWrapper(gym.Wrapper):
    """
    Inserts one black rectangle of configurable size into the environment.
    """
    def __init__(
        self,
        env: gym.Env,
        static: bool,
        min_patch_frac: float,
        max_patch_frac: float,
    ):
        """
        Args:
            env: The environment to wrap.
            static: If True, sample the patch region once and keep it fixed. If False,
                re-sample a new patch region on every frame.
            min_patch_frac: Minimum height/width fraction of the patch relative to image size.
            max_patch_frac: Maximum height/width fraction of the patch relative to image size.
        """
        super().__init__(env)
        if not (0.0 <= min_patch_frac <= max_patch_frac <= 1.0):
            raise ValueError(
                f"Invalid patch fractions: ({min_patch_frac}, {max_patch_frac}). "
                "Values must be in [0.0, 1.0] and min <= max."
            )
        self.static = static
        self.min_patch_frac = min_patch_frac
        self.max_patch_frac = max_patch_frac

        # Normalised rectangle coordinates: (top_frac, left_frac, height_frac, width_frac) from [0, 1]
        self.patch_frac: Optional[Tuple[float, float, float, float]] = None

    def _reset_patch_fraction(self):
        """
        Randomly select a rectangular region within the image to blackout and store its
        normalized coordinates and size in `self.patch_frac`.
        """
        patch_height_frac = random.uniform(self.min_patch_frac, self.max_patch_frac)
        patch_width_frac = random.uniform(self.min_patch_frac, self.max_patch_frac)

        top_frac = random.uniform(0.0, 1.0 - patch_height_frac)
        left_frac = random.uniform(0.0, 1.0 - patch_width_frac)

        self.patch_frac = (top_frac, left_frac, patch_height_frac, patch_width_frac)

    def _apply_patch(self, img: np.ndarray) -> np.ndarray:
        """
        Apply a black rectangle to the input image using a fixed patch region.
        
        Args:
            img: The input image to apply the patch to.

        Returns:
            The patched image.
        """
        if self.patch_frac is None or not self.static:
            # Initialize the patch region using the first image
            self._reset_patch_fraction()

        height, width = img.shape[:2]
        top_frac, left_frac, patch_height_frac, patch_width_frac = self.patch_frac
        
        patch_top = int(round(top_frac * height))
        patch_left = int(round(left_frac * width))
        patch_height = int(round(patch_height_frac * height))
        patch_width = int(round(patch_width_frac * width))
        
        img = img.copy()
        img[patch_top : patch_top + patch_height, patch_left : patch_left + patch_width] = 0
        return img

    def _patch_obs(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply the fixed black patch to each image in the observation.

        Args:
            observation (Dict[str, Any]): A dictionary containing environment observations.

        Returns:
            Dict[str, Any]: The observation dictionary with patched pixel data.
        """
        if "pixels" in observation:
            if isinstance(observation["pixels"], dict):  # Multi-camera
                for camera, img in observation["pixels"].items():
                    observation["pixels"][camera] = self._apply_patch(img.copy())
            else:  # Single-camera
                observation["pixels"] = self._apply_patch(observation["pixels"].copy())

        return observation


    # Gym API overrides
    def reset(self, **kwargs):
        # Call env.reset(...), then patch the returned obs
        obs, info = super().reset(**kwargs)
        return self._patch_obs(obs), info

    def step(self, action):
        # Call env.step(...), then patch the returned obs
        obs, rew, term, trunc, info = super().step(action)
        return self._patch_obs(obs), rew, term, trunc, info

    def render(self, *args, **kwargs):
        frame = super().render(*args, **kwargs)
        if frame is None:
            warnings.warn(
                "PerturbationWrapper: render() produced no image. "
                "Perturbation will not be visualized.",
                UserWarning,
                stacklevel=2,
            )
        else:
            frame = self._apply_patch(frame.copy())

        return frame
