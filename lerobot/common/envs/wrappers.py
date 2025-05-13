import numpy as np
import random
import warnings
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym



class PerturbationWrapper(gym.Wrapper):
    """
    Inserts one black rectangle of fixed size and position into the
    environment.
    """
    def __init__(
        self,
        env: gym.Env,
        min_patch_frac: float = 0.1,
        max_patch_frac: float = 0.2,
    ):
        """
        Args:
            env: The environment to wrap.
            min_patch_frac: Minimum height/width fraction of the patch relative to image size.
            max_patch_frac: Maximum height/width fraction of the patch relative to image size.
        """
        super().__init__(env)
        if not (0.0 <= min_patch_frac <= max_patch_frac <= 1.0):
            raise ValueError(
                f"Invalid patch fractions: ({min_patch_frac}, {max_patch_frac}). "
                "Values must be in [0.0, 1.0] and min <= max."
            )
        self.min_patch_frac = min_patch_frac
        self.max_patch_frac = max_patch_frac

        # Will hold coordinates of top-left corner as well as height and width once computed
        self.patch_region: Optional[Tuple[int,int,int,int]] = None

    def _init_patch_region(self, img: np.ndarray):
        """
        Randomly select a rectangular region within the image to blackout and store its
        coordinates and size in `self.patch_region`.

        Args:
            img: Sample image of shape (H, W, C) used to determine the patch region.
        """
        height, width = img.shape[:2]

        patch_height = random.randint(int(self.min_patch_frac * height), int(self.max_patch_frac * height))
        patch_width = random.randint(int(self.min_patch_frac * width), int(self.max_patch_frac * width))

        top = random.randint(0, height - patch_height)
        left = random.randint(0, width - patch_width)

        self.patch_region = (top, left, patch_height, patch_width)

    def _apply_patch(self, img: np.ndarray) -> np.ndarray:
        """
        Apply a black rectangle to the input image using a fixed patch region.
        
        Args:
            img: The input image to apply the patch to.

        Returns:
            The patched image.
        """
        if self.patch_region is None:
            # Initialize the patch region using the first image
            self._init_patch_region(img)

        top, left, patch_height, patch_width = self.patch_region
        img[top : top + patch_height, left : left + patch_width] = 0
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
        return frame
