import numpy as np
import matplotlib.pyplot as plt

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
