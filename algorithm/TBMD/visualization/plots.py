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
    zmin=None,  # Add zmin parameter
    zmax=None,   # Add zmax parameter
    wells=None,  # Скважины для отображения
    frame_step=1  # Отображать каждый N-й кадр (по умолчанию каждый)
):
    """
    Visualizes a tensor of images. If the tensor is 3D (H, W, T), each (H, W) slice is 
    treated as a grayscale frame. If the tensor is 4D (H, W, C, T), each slice is treated 
    as a color image.
    
    Parameters:
      - tensor (numpy.ndarray): The data to visualize. Expected shapes:
          * (H, W, T) for grayscale images.
          * (H, W, C, T) for color images.
      - subject_name (str): Title label for the subject.
      - save_path (str): If provided, the path where the plot will be saved.
      - cmap (str): Matplotlib colormap used for imshow when displaying grayscale images.
      - cols (int): Number of columns in the grid layout.
      - show_colorbar (bool): Whether to display colorbars for each subplot.
      - zmin (float, optional): Minimum value for the color scale (vmin in imshow). 
                                If None, calculated per frame as the minimum non-zero value.
      - zmax (float, optional): Maximum value for the color scale (vmax in imshow).
                                If None, calculated per frame as the maximum value.
      - wells (dict or list): Координаты скважин. Может быть:
                             * dict с ключами = subject_name и значениями = списки координат [(x1, y1), ...]
                             * dict с ключами = subject_name и значениями = вложенные словари {frame_idx: [(x1, y1), ...], ...}
                             * простой список координат [(x1, y1), ...] для всех кадров
      - frame_step (int): Шаг для отображения кадров. При frame_step=10 будет отображаться каждый 10-й кадр.
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

    # Выбираем кадры с заданным шагом
    frame_indices = list(range(0, T, frame_step))
    selected_frames_count = len(frame_indices)

    # Determine the number of rows needed for the grid layout.
    rows = (selected_frames_count + cols - 1) // cols

    # Увеличиваем размер фигуры для более крупных subplots
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 6.5 * rows))
    
    # Handle case where rows=1 or cols=1, making axes not 2D
    if selected_frames_count <= 1:
        axes = np.array([axes]) # Ensure axes is always iterable
    axes = axes.flatten()  # Flatten axes array for easier iteration.

    for display_idx, ax in enumerate(axes):
        if display_idx < selected_frames_count:
            # Получаем реальный индекс кадра
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
            
            # Отображение скважин
            if wells is not None:
                wells_to_plot = None
                # Определяем какие скважины отображать для данного subject/frame
                if subject_name is not None:
                    # Проверяем разные форматы wells
                    if isinstance(wells, dict):
                        wells_to_plot = wells.get(subject_name)
                        # Если wells[subject_name] - словарь с ключами-номерами кадров
                        if isinstance(wells_to_plot, dict):
                            wells_to_plot = wells_to_plot.get(frame_idx)
                        # Проверяем альтернативный ключ subject_name_frame_idx
                        elif wells_to_plot is None:
                            wells_to_plot = wells.get(f"{subject_name}_{frame_idx}")
                else:
                    # Если subject_name не указан, пытаемся использовать wells напрямую
                    if isinstance(wells, list) or (isinstance(wells, np.ndarray) and wells.ndim >= 2):
                        wells_to_plot = wells
                
                # Отображаем скважины, если нашли
                if wells_to_plot is not None and len(wells_to_plot) > 0:
                    wells_array = np.array(wells_to_plot)
                    
                    # Отображение скважин без масштабирования (координаты должны быть в пикселях)
                    ax.scatter(wells_array[:, 0], wells_array[:, 1], c='red', marker='o', s=80, label='Wells')
                    
                    ax.legend(loc='upper right', fontsize=18)
            
            if show_colorbar:
                 # For 3D tensors, colorbar uses calculated vmin/vmax
                 # For 4D tensors, colorbar might not be meaningful unless specific channel shown
                 if tensor.ndim == 3:
                     cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pressure, psi")
                     cbar.ax.tick_params(labelsize=16)
                     cbar.set_label("Pressure, psi", size=18)                 # else: # Optional: add colorbar logic for 4D if needed
                 #    pass 
        ax.axis("off")

    # Adjust title position if it exists and make layout more compact
    if subject_name:
        fig.suptitle(f"Subject: {subject_name}", fontsize=40)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to prevent title overlap
        # Уменьшаем расстояние между изображениями для более компактного вида
        plt.subplots_adjust(wspace=0.001, hspace=0.15)
    else:
        plt.tight_layout()
        # Уменьшаем расстояние между изображениями для более компактного вида
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
    figsize: tuple = (20, 6),
    wspace: float = 0.05,
    hspace: float = 0.05,
    titles: tuple = ("Original", "Reconstructed", "Difference"),
    colorbar_labels: tuple = ("", "", ""),
    show_colorbar: bool = True,
    colorbar_fontsize: int = 12,
    separate_figures: bool = False,
    return_figures: bool = False,
):
    """
    Visualise *original*, *reconstructed* matrices and their difference.
    
    Универсальная функция для отображения оригинальной, реконструированной
    матриц и их разницы. Поддерживает как объединённый, так и раздельный вывод.

    Parameters
    ----------
    original : np.ndarray | torch.Tensor
        Ground-truth 2-D data.
    reconstructed : np.ndarray | torch.Tensor
        Reconstructed (or predicted) 2-D data of the same shape as *original*.
    common_cmap : str, default "viridis"
        Colormap shared by the first two panels.
    diff_cmap : str, default "RdBu_r"
        Diverging colormap for the difference panel (centered at 0).
    figsize : tuple, default (20, 6)
        Figure size. For combined: full width. For separate: each panel gets figsize[0]/3.
    wspace : float, default 0.05
        Horizontal space between subplots (for combined mode).
    hspace : float, default 0.05
        Vertical space between subplots (for combined mode).
    titles : tuple(str, str, str), default ("Original", "Reconstructed", "Difference")
        Panel titles for (original, reconstructed, difference). Use empty strings to hide.
    colorbar_labels : tuple(str, str, str), default ("", "", "")
        Labels for colorbars (e.g., ("Pressure, psi", "Pressure, psi", "Pressure diff, psi")).
    show_colorbar : bool, default True
        Whether to attach colour bars to every panel.
    colorbar_fontsize : int, default 12
        Font size for colorbar labels and ticks.
    separate_figures : bool, default False
        If True, creates three separate figures instead of one combined.
    return_figures : bool, default False
        If True, returns figure(s) instead of showing them. Useful for saving.

    Returns
    -------
    None or figure(s)
        If return_figures=True:
            - Combined mode: returns single Figure
            - Separate mode: returns tuple of 3 Figures (fig_orig, fig_rec, fig_diff)

    Examples
    --------
    >>> # Combined figure (default)
    >>> plot_original_reconstructed_diff(orig, rec)
    
    >>> # Separate figures with custom labels
    >>> plot_original_reconstructed_diff(
    ...     orig, rec,
    ...     separate_figures=True,
    ...     colorbar_labels=("Pressure, psi", "Pressure, psi", "Pressure diff, psi")
    ... )
    
    >>> # Get figures for saving
    >>> figs = plot_original_reconstructed_diff(orig, rec, separate_figures=True, return_figures=True)
    >>> figs[0].savefig("original.png", dpi=300, bbox_inches='tight')
    """
    import numpy as _np
    import torch as _torch
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # ---------- Helpers ----------
    def _to_numpy(x):
        """Convert tensor/array to numpy."""
        if _torch.is_tensor(x):
            return x.detach().cpu().numpy()
        elif isinstance(x, _np.ndarray):
            return x
        else:
            raise TypeError("Input must be a NumPy array or a torch.Tensor.")

    def _safe_positive_min(arr: _np.ndarray):
        """Get minimum positive value or None if no positive values."""
        pos = arr[arr > 0]
        return float(pos.min()) if pos.size > 0 else None

    def _ensure_tuple(val, length=3, default=""):
        """Ensure val is a tuple of given length."""
        if val is None:
            return tuple([default] * length)
        if len(val) != length:
            raise ValueError(f"Expected tuple of length {length}, got {len(val)}")
        return tuple(val)

    def _add_colorbar(ax, im, label="", fontsize=12):
        """Add colorbar aligned to axes height."""
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        if label:
            cbar.set_label(label, fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize - 2)
        return cbar

    # ---------- Data Preparation ----------
    orig_np = _to_numpy(original)
    rec_np = _to_numpy(reconstructed)

    if orig_np.shape != rec_np.shape:
        raise ValueError(
            f"Original and reconstructed shapes differ: {orig_np.shape} vs {rec_np.shape}."
        )

    diff_np = orig_np - rec_np

    # ---------- Color Scale Limits ----------
    # Common vmin/vmax for original & reconstructed
    pos_min_orig = _safe_positive_min(orig_np)
    pos_min_rec = _safe_positive_min(rec_np)
    
    if pos_min_orig is not None and pos_min_rec is not None:
        vmin_common = min(pos_min_orig, pos_min_rec)
    elif pos_min_orig is not None:
        vmin_common = pos_min_orig
    elif pos_min_rec is not None:
        vmin_common = pos_min_rec
    else:
        vmin_common = float(min(orig_np.min(), rec_np.min()))

    vmax_common = float(max(orig_np.max(), rec_np.max()))
    
    # Ensure vmax > vmin
    if vmax_common <= vmin_common:
        vmax_common = 1.0 if vmin_common == 0 else vmin_common * 1.1

    # Symmetric limits for difference (centered at 0)
    d_abs_max = float(max(abs(diff_np.min()), abs(diff_np.max())))
    if d_abs_max == 0:
        d_abs_max = 1.0  # Avoid zero range
    vmin_diff, vmax_diff = -d_abs_max, d_abs_max

    # Ensure tuples
    titles = _ensure_tuple(titles, 3, "")
    colorbar_labels = _ensure_tuple(colorbar_labels, 3, "")

    # ---------- Plotting ----------
    if separate_figures:
        # Create three separate figures
        panel_width = figsize[0] / 3
        panel_height = figsize[1]
        
        figures = []
        data_configs = [
            (orig_np, common_cmap, vmin_common, vmax_common, titles[0], colorbar_labels[0]),
            (rec_np, common_cmap, vmin_common, vmax_common, titles[1], colorbar_labels[1]),
            (diff_np, diff_cmap, vmin_diff, vmax_diff, titles[2], colorbar_labels[2]),
        ]
        
        for data, cmap, vmin, vmax, title, cbar_label in data_configs:
            fig, ax = plt.subplots(figsize=(panel_width, panel_height))
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
            ax.axis('off')
            if title:
                ax.set_title(title, fontsize=colorbar_fontsize + 2)
            if show_colorbar:
                _add_colorbar(ax, im, label=cbar_label, fontsize=colorbar_fontsize)
            fig.tight_layout()
            figures.append(fig)
            
            if not return_figures:
                plt.show()
        
        if return_figures:
            return tuple(figures)
    
    else:
        # Combined figure with 3 panels
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.subplots_adjust(wspace=wspace, hspace=hspace)

        im0 = axes[0].imshow(orig_np, cmap=common_cmap, vmin=vmin_common, vmax=vmax_common, aspect='equal')
        im1 = axes[1].imshow(rec_np, cmap=common_cmap, vmin=vmin_common, vmax=vmax_common, aspect='equal')
        im2 = axes[2].imshow(diff_np, cmap=diff_cmap, vmin=vmin_diff, vmax=vmax_diff, aspect='equal')

        for ax, title in zip(axes, titles):
            ax.axis('off')
            if title:
                ax.set_title(title, fontsize=colorbar_fontsize + 2)

        if show_colorbar:
            for ax, im, label in zip(axes, (im0, im1, im2), colorbar_labels):
                _add_colorbar(ax, im, label=label, fontsize=colorbar_fontsize)

        fig.tight_layout()
        
        if return_figures:
            return fig
        else:
            plt.show()


def visualize_wells_placement(wells_matrix, title="Wells placement"):
    """
    Visualizes wells placement matrix.
    
    Args:
        wells_matrix: Binary tensor with 1s at well positions
        title: Title for the plot
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