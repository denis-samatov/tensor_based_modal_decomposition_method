import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split




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
                images_list,
                test_size=1 - train_ratio,
                random_state=experiment_id,
                shuffle=shuffle
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
