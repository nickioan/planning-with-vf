"""Dataset files written by `scripts/collect_data.py` and the normalizers fit on them.

Each collection run writes one `data.npz` with one entry per 4-footstep plan:

    plans            (N, 4, 6)  footsteps relative to the character, (x, y, z, x_tilt, y_tilt, label).
                                For flat/platform data x, y are in the heading-to-waypoint frame
                                (see `vfplan.geometry.glob_to_rel`); for hurdle data they are world-axis
                                offsets from the root. `label` is 1 if the plan was executed without
                                falling, otherwise a value < 1.
    humanoid_state   (N, 56)    character state when the plan was made (see `policy_state`).
    returns          (N,)       discounted return of the plan (gamma = 0.75); not for flat data.
    heightmaps       (N, 32, 32) egocentric heightfield minus the root height (flat/platform).
    waypoints        (N, 3)     waypoint minus root position (flat/platform).
    hurdle_states    (N, 4)     next hurdle minus root position, and its yaw (hurdle).
"""

import glob
import os

import numpy as np
import torch


class GaussianNormalizer:
    """Normalizes to zero mean and unit variance over all but the last axis."""

    def __init__(self, means, stds):
        self.means = np.asarray(means)
        self.stds = np.asarray(stds)

    @classmethod
    def fit(cls, X):
        axis = tuple(range(len(X.shape) - 1))
        return cls(X.mean(axis=axis), X.std(axis=axis))

    def normalize(self, x):
        return (x - self.means) / self.stds

    def unnormalize(self, x):
        return x * self.stds + self.means

    def state_dict(self):
        return {"means": torch.as_tensor(self.means), "stds": torch.as_tensor(self.stds)}

    @classmethod
    def from_state_dict(cls, d):
        return cls(d["means"].numpy(), d["stds"].numpy())


def find_data_files(paths):
    """Expands directories (searched recursively for data.npz) and globs into file paths."""
    files = []
    for path in paths:
        if os.path.isdir(path):
            files += sorted(glob.glob(os.path.join(path, "**", "data.npz"), recursive=True))
        else:
            files += sorted(glob.glob(path))
    if not files:
        raise FileNotFoundError(f"No data.npz files found in {paths}")
    return files


def load_datasets(paths, keys):
    """Loads and concatenates `keys` from every data.npz found under `paths`."""
    files = find_data_files(paths)
    data = {k: [] for k in keys}
    for f in files:
        with np.load(f) as d:
            for k in keys:
                data[k].append(d[k])
    print(f"Loaded {len(files)} file(s): {files}")
    return {k: np.concatenate(v, axis=0) for k, v in data.items()}
