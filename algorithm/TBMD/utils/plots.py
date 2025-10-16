import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager
from typing import Optional, Union, Dict, List, Tuple, Any
import warnings


def visualize_tensor(
    tensor,
    subject_name=None,
    save_path=None,
    cmap="gray",
    cols=5,
    show_colorbar=False,
    zmin=None,
    zmax=None,
    wells=None,
    frame_step=1
):
    """Visualize a tensor of images.

    This function visualizes a tensor of images. If the tensor is 3D (H, W, T),
    each (H, W) slice is treated as a grayscale frame. If the tensor is 4D
    (H, W, C, T), each slice is treated as a color image.

    Parameters
    ----------
    tensor : numpy.ndarray
        The data to visualize. Expected shapes are (H, W, T) for grayscale
        images or (H, W, C, T) for color images.
    subject_name : str, optional
        The title label for the subject, by default None.
    save_path : str, optional
        The path where the plot will be saved, by default None.
    cmap : str, optional
        The Matplotlib colormap for grayscale images, by default "gray".
    cols : int, optional
        The number of columns in the grid layout, by default 5.
    show_colorbar : bool, optional
        Whether to display colorbars for each subplot, by default False.
    zmin : float, optional
        The minimum value for the color scale (vmin in imshow). If None, it is
        calculated per frame as the minimum non-zero value, by default None.
    zmax : float, optional
        The maximum value for the color scale (vmax in imshow). If None, it is
        calculated per frame as the maximum value, by default None.
    wells : dict or list, optional
        The coordinates of wells to display. Can be a dictionary with subject
        names as keys and lists of coordinates as values, a dictionary with
        subject names as keys and nested dictionaries with frame indices as
        keys, or a simple list of coordinates for all frames, by default None.
    frame_step : int, optional
        The step for displaying frames. For example, a `frame_step` of 10
        will display every 10th frame, by default 1.
    """
    # Ensure tensor has either 3 or 4 dimensions.
    if tensor.ndim not in (3, 4):
        raise ValueError(f"Expected tensor with 3 or 4 dimensions, got shape {tensor.shape}.")

    # Optional: Check pixel intensity range.
    # Note: This check might conflict with user-provided zmin/zmax. Consider adjusting or removing.
    # if tensor.min() < 0 or tensor.max() > 1:
    #     print("Warning: Tensor values are outside the range [0, 1]. Verify if correct.")

    # Determine the number of frames (T) and image shape.
    if tensor.ndim == 3:
        H, W, T = tensor.shape
    else:  # tensor.ndim == 4
        H, W, C, T = tensor.shape

    # Select frames with the specified step
    frame_indices = list(range(0, T, frame_step))
    selected_frames_count = len(frame_indices)

    # Determine the number of rows needed for the grid layout.
    rows = (selected_frames_count + cols - 1) // cols

    # Increase the figure size for larger subplots
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5.5 * rows))
    
    # Handle case where rows=1 or cols=1, making axes not 2D
    if selected_frames_count <= 1:
        axes = np.array([axes]) # Ensure axes is always iterable
    axes = axes.flatten()  # Flatten axes array for easier iteration.

    for display_idx, ax in enumerate(axes):
        if display_idx < selected_frames_count:
            # Get the actual frame index
            frame_idx = frame_indices[display_idx]
            
            # Extract the frame depending on tensor dimensions.
            if tensor.ndim == 3:
                frame = tensor[:, :, frame_idx]
                
                # Determine vmin and vmax for this frame
                current_zmin = zmin
                if current_zmin is None:
                    # Calculate min non-zero value, default to 0 if all zero
                    non_zero_frame = frame[frame != 0]
                    if non_zero_frame.size > 0:
                        current_zmin = non_zero_frame.min()
                    else:
                         current_zmin = 0 # Frame is all zeros

                current_zmax = zmax
                if current_zmax is None:
                    current_zmax = frame.max()
                    # Handle case where max is 0 (or less than min)
                    if current_zmax <= current_zmin and current_zmax == 0:
                         current_zmax = 1 # Avoid vmin=vmax=0 if possible, adjust as needed

                im = ax.imshow(frame, cmap=cmap, aspect="equal", vmin=current_zmin, vmax=current_zmax)
            else:  # 4D tensor: display color image.
                # vmin/vmax typically not used directly for RGB images with imshow
                frame = tensor[:, :, :, frame_idx]
                # Ensure frame data is in displayable range [0,1] or [0,255] if needed
                # frame = np.clip(frame, 0, 1) # Example clipping if data is float [0,1]
                im = ax.imshow(frame, aspect="equal") 
                
            ax.set_title(f"Frame {frame_idx + 1}", fontsize=20)
            
            # Display wells
            if wells is not None:
                wells_to_plot = None
                # Determine which wells to display for the given subject/frame
                if subject_name is not None:
                    # Check different formats of wells
                    if isinstance(wells, dict):
                        wells_to_plot = wells.get(subject_name)
                        # If wells[subject_name] is a dictionary with frame numbers as keys
                        if isinstance(wells_to_plot, dict):
                            wells_to_plot = wells_to_plot.get(frame_idx)
                        # Check for an alternative key subject_name_frame_idx
                        elif wells_to_plot is None:
                            wells_to_plot = wells.get(f"{subject_name}_{frame_idx}")
                else:
                    # If subject_name is not specified, try to use wells directly
                    if isinstance(wells, list) or (isinstance(wells, np.ndarray) and wells.ndim >= 2):
                        wells_to_plot = wells
                
                # Display wells if found
                if wells_to_plot is not None and len(wells_to_plot) > 0:
                    wells_array = np.array(wells_to_plot)
                    
                    # Display wells without scaling (coordinates should be in pixels)
                    ax.scatter(wells_array[:, 0], wells_array[:, 1], c='red', marker='o', s=80, label='Wells')
                    
                    ax.legend(loc='upper right', fontsize=15)
            
            if show_colorbar:
                 # For 3D tensors, colorbar uses calculated vmin/vmax
                 # For 4D tensors, colorbar might not be meaningful unless specific channel shown
                 if tensor.ndim == 3:
                     fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                 # else: # Optional: add colorbar logic for 4D if needed
                 #    pass 
        ax.axis("off")

    # Adjust title position if it exists and make layout more compact
    if subject_name:
        fig.suptitle(f"Subject: {subject_name}", fontsize=40)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to prevent title overlap
        # Reduce the distance between images for a more compact view
        plt.subplots_adjust(wspace=0.001, hspace=0.15)
    else:
        plt.tight_layout()
        # Reduce the distance between images for a more compact view
        plt.subplots_adjust(wspace=0.001, hspace=0.15)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_two_matrices(
    X,
    Y,
    zmin: float = None,
    zmax: float = None,
    cmap: str = 'viridis',
    figsize: tuple = (12, 6),
    titles: tuple = ('Matrix X', 'Matrix Y'),
    show_colorbar: bool = True
):
    """
    Plot two 2D arrays or tensors side by side with a shared colormap scale.

    Parameters
    ----------
    X : numpy.ndarray or torch.Tensor
        First matrix to visualize. Can be on CPU or GPU.
    Y : numpy.ndarray or torch.Tensor
        Second matrix to visualize.
    zmin : float, optional
        Lower bound for color scale. If None, uses the smallest non-zero value across both matrices.
    zmax : float, optional
        Upper bound for color scale. If None, uses the maximum value across both matrices.
    cmap : str, default 'viridis'
        Name of the Matplotlib colormap to use.
    figsize : tuple of two ints, default (12, 6)
        Size of the entire figure in inches (width, height).
    titles : tuple of str, default ('Matrix X', 'Matrix Y')
        Titles for the two subplots.
    show_colorbar : bool, default True
        If True, show colorbar for each subplot.

    Example
    -------
    plot_two_matrices(X, Y, zmin=0.1, zmax=1.0, cmap='plasma', figsize=(10, 5), titles=('Input', 'Output'))
    """
    def _get_nonzero_min(frame):
        """Get the minimum non-zero value from a tensor or array."""
        if hasattr(frame, 'numpy'):
            non_zero = frame[frame > 0]  # Only positive values
            return non_zero.min().item() if non_zero.numel() > 0 else None
        else:
            non_zero = frame[frame > 0]  # Only positive values
            return non_zero.min() if non_zero.size > 0 else None

    def _get_max(frame):
        """Get the maximum value from a tensor or array."""
        return frame.max().item() if hasattr(frame, 'numpy') else frame.max()

    # Compute shared vmin and vmax across both matrices
    if zmin is not None:
        vmin = zmin
    else:
        # Find minimum non-zero value across both matrices
        min_X = _get_nonzero_min(X)
        min_Y = _get_nonzero_min(Y)
        
        # Select the smallest non-zero value between both matrices
        if min_X is not None and min_Y is not None:
            vmin = min(min_X, min_Y)
        elif min_X is not None:
            vmin = min_X
        elif min_Y is not None:
            vmin = min_Y
        else:
            # If no positive values found, default to 0
            vmin = 0
            
    if zmax is not None:
        vmax = zmax
    else:
        vmax = max(_get_max(X), _get_max(Y))
    
    # Ensure vmax > vmin, especially if vmin is 0
    if vmax <= vmin:
        if vmin == 0:
            vmax = 1  # Default when all values are 0
        else:
            vmax = vmin * 1.1  # Add 10% to vmin if vmin > 0

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, frame, title in zip(axes, (X, Y), titles):
        # Convert to numpy if it's a tensor
        array = frame.cpu().numpy() if hasattr(frame, 'cpu') else np.array(frame)
        
        im = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
        
        if show_colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
        ax.set_title(title)

    plt.tight_layout()
    plt.show()


def normalize_for_rgb_display(data):
    """Normalize data for RGB display.

    This is a utility function to normalize data for RGB display to avoid
    matplotlib warnings.

    Parameters
    ----------
    data : numpy.ndarray or torch.Tensor
        The data to normalize.

    Returns
    -------
    numpy.ndarray
        The normalized data in the range [0, 1].

    Examples
    --------
    >>> normalized_data = normalize_for_rgb_display(reconstruction_data)
    >>> plt.imshow(normalized_data)
    """
    # Convert to numpy if it's a tensor
    array = data.cpu().numpy() if hasattr(data, 'cpu') else np.array(data)
    
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


def plot_original_reconstructed_diff(
    original,
    reconstructed,
    *,
    common_cmap: str = "viridis",
    diff_cmap: str = "RdBu_r",
    figsize: tuple = (12, 4),
    titles: tuple = ("Original", "Reconstructed", "Difference"),
    show_colorbar: bool = True,
):
    """Visualise *original*, *reconstructed* matrices and their difference.

    Parameters
    ----------
    original : np.ndarray | torch.Tensor
        Ground-truth 2-D data.
    reconstructed : np.ndarray | torch.Tensor
        Reconstructed (or predicted) 2-D data of the same shape as *original*.
    common_cmap : str, default "viridis"
        Colormap shared by the first two panels.
    diff_cmap : str, default "RdBu_r"
        Diverging colormap for the difference panel.
    figsize : tuple, default (12, 4)
        Figure size passed to ``plt.subplots``.
    titles : tuple(str, str, str)
        Panel titles for (original, reconstructed, difference).
    show_colorbar : bool, default True
        Whether to attach colour bars to every panel.

    Notes
    -----
    *If* the inputs reside on GPU, they are moved to CPU **without** cloning
    (no gradients preserved).
    """
    import numpy as _np  # local import to prevent polluting public namespace
    import torch as _torch

    # Utility: convert to CPU numpy for safe matplotlib handling
    def _to_numpy(x):
        if _torch.is_tensor(x):
            return x.detach().cpu().numpy()
        elif isinstance(x, _np.ndarray):
            return x
        else:
            raise TypeError("Input must be a NumPy array or a torch.Tensor.")

    def _get_nonzero_min(frame):
        """Get the minimum non-zero value from a tensor or array."""
        if hasattr(frame, 'numpy'):
            non_zero = frame[frame > 0]  # Only positive values
            return non_zero.min().item() if non_zero.numel() > 0 else None
        else:
            non_zero = frame[frame > 0]  # Only positive values
            return non_zero.min() if non_zero.size > 0 else None

    def _get_max(frame):
        """Get the maximum value from a tensor or array."""
        return frame.max().item() if hasattr(frame, 'numpy') else frame.max()

    orig_np = _to_numpy(original)
    rec_np = _to_numpy(reconstructed)

    if orig_np.shape != rec_np.shape:
        raise ValueError(
            f"Original and reconstructed shapes differ: {orig_np.shape} vs {rec_np.shape}."
        )

    diff_np = orig_np - rec_np

    # Compute shared colour scale for the first two images using helper functions
    min_orig = _get_nonzero_min(orig_np)
    min_rec = _get_nonzero_min(rec_np)
    
    # Select the smallest non-zero value between both matrices
    if min_orig is not None and min_rec is not None:
        vmin_common = min(min_orig, min_rec)
    elif min_orig is not None:
        vmin_common = min_orig
    elif min_rec is not None:
        vmin_common = min_rec
    else:
        # If no positive values found, default to 0
        vmin_common = 0

    vmax_common = max(_get_max(orig_np), _get_max(rec_np))
    
    # Ensure vmax > vmin, especially if vmin is 0
    if vmax_common <= vmin_common:
        if vmin_common == 0:
            vmax_common = 1  # Default when all values are 0
        else:
            vmax_common = vmin_common * 1.1  # Add 10% to vmin if vmin > 0

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    im0 = axes[0].imshow(orig_np, cmap=common_cmap, vmin=vmin_common, vmax=vmax_common)
    axes[0].set_title(titles[0])

    im1 = axes[1].imshow(rec_np, cmap=common_cmap, vmin=vmin_common, vmax=vmax_common)
    axes[1].set_title(titles[1])

    im2 = axes[2].imshow(diff_np, cmap=diff_cmap)
    axes[2].set_title(titles[2])

    if show_colorbar:
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

def visualize_wells_placement(wells_matrix, title="Wells placement"):
    """Visualize the placement of wells.

    This function visualizes a wells placement matrix, where 1s indicate the
    positions of wells.

    Parameters
    ----------
    wells_matrix : torch.Tensor
        A binary tensor with 1s at the well positions.
    title : str, optional
        The title for the plot, by default "Wells placement".
    """
    wells_np = wells_matrix.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(wells_np.shape[1] / 10, wells_np.shape[0] / 10))
    ax.set_facecolor("black")
    ax.imshow(np.zeros(wells_np.shape), cmap="gray", origin="upper")
    pos = np.argwhere(wells_np == 1)
    if pos.size > 0:
        ax.scatter(pos[:, 1], pos[:, 0], s=50, c="blue", marker="o", alpha=0.8, label="Sensors")
        ax.legend()  # Add this line to display the legend
    ax.set_title(title, color="white")
    ax.axis("off")
    plt.show()