import logging
from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logger = logging.getLogger(__name__)

# =================================================================================================
# EXISTING CLASSES FROM data/splitters.py
# =================================================================================================


class DataSplitter:
    """
    Split data into train, validation, and test subsets.

    Examples:
        >>> splitter = DataSplitter(train_ratio=0.7, val_ratio=0.15)
        >>> train, val, test = splitter.split(tensor)
    """

    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: Optional[float] = None,
        shuffle: bool = True,
        seed: Optional[int] = 42,
    ):
        """
        Args:
            train_ratio: Fraction of data used for training.
            val_ratio: Fraction of data used for validation.
            test_ratio: Fraction of data used for testing; computed automatically if None.
            shuffle: Whether to shuffle data before splitting.
            seed: Random seed for reproducibility.
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        if test_ratio is None:
            self.test_ratio = 1.0 - train_ratio - val_ratio
        else:
            self.test_ratio = test_ratio

        # Validation
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"split ratios must sum to 1.0, got: {total}")

        if any(r < 0 for r in [self.train_ratio, self.val_ratio, self.test_ratio]):
            raise ValueError("all split ratios must be non-negative")

        self.shuffle = shuffle
        self.seed = seed

    def split(
        self, data: torch.Tensor, split_dim: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split data.

        Args:
            data: Input data.
            split_dim: Dimension along which to split, usually time.

        Returns:
            (train_data, val_data, test_data)
        """
        n_samples = data.shape[split_dim]

        # Compute split sizes
        n_train = int(n_samples * self.train_ratio)
        n_val = int(n_samples * self.val_ratio)
        n_test = n_samples - n_train - n_val

        logger.info(f"Split sizes: train={n_train}, val={n_val}, test={n_test}")

        # Indices
        indices = torch.arange(n_samples)

        if self.shuffle:
            if self.seed is not None:
                torch.manual_seed(self.seed)
            perm = torch.randperm(n_samples)
            indices = indices[perm]

        # Split indices
        train_indices = indices[:n_train]
        val_indices = indices[n_train : n_train + n_val]
        test_indices = indices[n_train + n_val :]

        # Extract data
        train_data = torch.index_select(data, split_dim, train_indices)
        val_data = torch.index_select(data, split_dim, val_indices)
        test_data = torch.index_select(data, split_dim, test_indices)

        return train_data, val_data, test_data

    def split_temporal(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split temporal data without shuffling.

        Useful for time series where order matters.

        Args:
            data: Temporal data.

        Returns:
            (train_data, val_data, test_data)
        """
        old_shuffle = self.shuffle
        self.shuffle = False

        result = self.split(data, split_dim=-1)

        self.shuffle = old_shuffle
        return result


# =================================================================================================
# MIGRATED FUNCTIONS FROM utils/split_data.py
# =================================================================================================


def split_data_in_memory(subject_images, num_experiments, train_ratio=0.8, shuffle=True):
    """Splits a stacked image tensor into training and testing sets.

    This function performs a random split for multiple experiments. For each
    experiment, it converts the subject's images into a list, uses
    `train_test_split` to divide them into training and testing sets, and
    stacks the resulting lists back into NumPy arrays.

    Args:
        subject_images (dict): A dictionary mapping subject IDs to NumPy arrays
            of shape (H, W, N), where N is the number of images.
        num_experiments (int): The number of random splits to perform.
        train_ratio (float, optional): The fraction of images to use for
            training. Defaults to 0.8.
        shuffle (bool, optional): Whether to shuffle the images before
            splitting. If `False`, a sequential split is used. Defaults to
            True.

    Returns:
        dict: A dictionary mapping each experiment ID to a dictionary with
        'train' and 'test' keys, which in turn map subject IDs to their
        corresponding data splits.
    """
    experiments_data = {}

    for experiment_id in tqdm(range(1, num_experiments + 1), desc="Experiments processed"):
        train_data = {}
        test_data = {}
        for subject, images in subject_images.items():
            # Check that images exist and have at least one image along the third dimension
            if images is None or images.shape[-1] < 1:
                print(f"Warning: Subject {subject} has no images.")
                continue

            # Convert the stacked images (H, W, N) into a list of N individual images (each of shape H, W)
            images_list = [images[..., i] for i in range(images.shape[-1])]

            # Perform the train/test split using the current experiment_id as the random seed.
            train_images, test_images = train_test_split(
                images_list, test_size=1 - train_ratio, random_state=experiment_id, shuffle=shuffle
            )

            # Stack back the lists into arrays along a new third dimension (if non-empty).
            train_data[subject] = np.stack(train_images, axis=-1) if train_images else None
            test_data[subject] = np.stack(test_images, axis=-1) if test_images else None

        experiments_data[experiment_id] = {"train": train_data, "test": test_data}

    return experiments_data


def split_data_in_memory_ordered(subject_images, train_ratio=0.8):
    """Splits a stacked image tensor into training and testing sets sequentially.

    The first `train_ratio` fraction of images is used for training, and the
    remaining images are used for testing. The images are expected to be in a
    NumPy array of shape (H, W, N), where N is the number of images.

    Args:
        subject_images (defaultdict): A dictionary mapping subject IDs to NumPy
            arrays of shape (H, W, N).
        train_ratio (float, optional): The fraction of images to use for
            training. Defaults to 0.8.

    Returns:
        tuple: A tuple containing two dictionaries, `train_data` and
        `test_data`, which map subject IDs to their corresponding data splits.
    """
    train_data = defaultdict(lambda: None)
    test_data = defaultdict(lambda: None)

    for subject, images in tqdm(subject_images.items(), desc="Experiments processed"):
        # images should be a numpy array with shape (H, W, N)
        if images is None:
            print(f"Warning: no images for subject {subject}.")
            continue

        n = images.shape[-1]
        if n == 0:
            print(f"Warning: no images for subject {subject}.")
            continue

        split_index = int(n * train_ratio)
        train_images = images[..., :split_index]
        test_images = images[..., split_index:]

        train_data[subject] = train_images
        test_data[subject] = test_images

        if not train_data and not test_data:
            raise ValueError("Experiment 1 does not contain any train or test data.")

    return train_data, test_data
