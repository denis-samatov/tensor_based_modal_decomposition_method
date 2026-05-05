import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, Dict, List, Tuple
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
                    if hasattr(non_zero_frame, 'numel') and non_zero_frame.numel() > 0:
                        current_zmin = non_zero_frame.min()
                    elif not hasattr(non_zero_frame, 'numel') and non_zero_frame.size > 0:
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
                     cbar.set_label("Pressure, psi", size=18)
                 # else: # Optional: add colorbar logic for 4D if needed
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
