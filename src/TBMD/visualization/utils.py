import numpy as np


def normalize_for_rgb_display(data):
    """
    Utility function to normalize data for RGB display to avoid matplotlib warnings.

    Parameters
    ----------
    data : numpy.ndarray or torch.Tensor
        Data to normalize

    Returns
    -------
    numpy.ndarray
        Normalized data in range [0, 1]

    Example
    -------
    >>> normalized_data = normalize_for_rgb_display(reconstruction_data)
    >>> plt.imshow(normalized_data)
    """
    # Convert to numpy if it's a tensor
    array = data.cpu().numpy() if hasattr(data, "cpu") else np.array(data)

    # Normalize to [0, 1] range
    array_min = array.min()
    array_max = array.max()

    if array_max > array_min:
        array = (array - array_min) / (array_max - array_min)
    else:
        # Handle case where all values are the same
        array = np.zeros_like(array)

    # Ensure data is properly clipped to [0, 1]
    array = np.clip(array, 0, 1)

    return array
