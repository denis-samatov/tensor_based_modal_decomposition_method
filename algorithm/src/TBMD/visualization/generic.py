import numpy as np
import matplotlib.pyplot as plt

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
