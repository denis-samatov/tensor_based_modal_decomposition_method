from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_analytics(
    df: pd.DataFrame,
    metrics: List[str] = ["error", "ssim", "psnr"],
    plot_type: str = "individual",
    title_prefix: str = "Experiment Results",
    figsize: Tuple[int, int] = (8, 5),
    save_path: Optional[str] = None,
    show_plots: bool = True,
) -> None:
    """
    Plot analytics results from DataFrame with comprehensive visualization options.

    Replicates the functionality of the original plot_analytics function from plots.py
    but adapted for DataFrame input format.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame from ExperimentRunner methods.
    metrics : List[str]
        Metrics to plot. Default: ['error', 'ssim', 'psnr']
    plot_type : str
        Type of plot: 'individual', 'combined', 'normalized', 'all'
        - 'individual': Separate plots for each metric
        - 'combined': All metrics on one plot (non-normalized)
        - 'normalized': Normalized Error (inverted) and SSIM
        - 'all': All above plot types
    title_prefix : str
        Prefix for plot titles.
    figsize : Tuple[int, int]
        Figure size for individual plots.
    save_path : str, optional
        Base path to save plots (will add suffixes for multiple plots).
    show_plots : bool
        Whether to display plots.
    """
    # Determine data format (with or without confidence intervals)
    any(f"{metric}_ci_lower" in df.columns for metric in metrics)
    has_mean_std = any(f"{metric}_mean" in df.columns for metric in metrics)

    # Extract data for plotting
    sensor_values = df["sensors"].values
    plot_data = {}

    for metric in metrics:
        if has_mean_std and f"{metric}_mean" in df.columns:
            # Data with confidence intervals
            plot_data[metric] = {
                "means": df[f"{metric}_mean"].values,
                "lower": df[f"{metric}_ci_lower"].values
                if f"{metric}_ci_lower" in df.columns
                else df[f"{metric}_mean"].values - df[f"{metric}_std"].values,
                "upper": df[f"{metric}_ci_upper"].values
                if f"{metric}_ci_upper" in df.columns
                else df[f"{metric}_mean"].values + df[f"{metric}_std"].values,
                "std": df[f"{metric}_std"].values if f"{metric}_std" in df.columns else None,
            }
        elif metric in df.columns:
            # Simple data without confidence intervals
            plot_data[metric] = {
                "means": df[metric].values,
                "lower": df[metric].values,  # No CI, use same values
                "upper": df[metric].values,
                "std": None,
            }
        else:
            print(f"Warning: Metric '{metric}' not found in DataFrame")
            continue

    if not plot_data:
        print("No valid metrics found in DataFrame")
        return

    # Color mapping
    colors = {"error": "blue", "ssim": "green", "psnr": "red", "mse": "orange"}

    def save_plot(suffix=""):
        if save_path:
            path = f"{save_path}_{suffix}.png" if suffix else f"{save_path}.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")

    # Plot 1: Individual plots for each metric
    if plot_type in ["individual", "all"]:
        for metric in plot_data.keys():
            data = plot_data[metric]
            color = colors.get(metric, "black")

            plt.figure(figsize=figsize)
            plt.plot(sensor_values, data["means"], color=color, label=f"Mean {metric.upper()}")
            plt.scatter(sensor_values, data["means"], color=color, marker="o", s=30)

            # Add confidence intervals or std deviation
            if not np.array_equal(data["lower"], data["means"]) or not np.array_equal(
                data["upper"], data["means"]
            ):
                plt.fill_between(
                    sensor_values,
                    data["lower"],
                    data["upper"],
                    color=color,
                    alpha=0.2,
                    label="95% CI",
                )

            plt.title(f"{metric.upper()} vs. Sensors")
            plt.xlabel("Number of Sensors (N)")

            if metric == "error":
                plt.ylabel("Error")
            elif metric == "ssim":
                plt.ylabel("SSIM")
            elif metric == "psnr":
                plt.ylabel("PSNR (dB)")
            else:
                plt.ylabel(metric.upper())

            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            if save_path:
                save_plot(f"{metric}")

            if show_plots:
                plt.show()
            else:
                plt.close()

    # Plot 2: Combined Normalized Plot (Error inverted and SSIM)
    if plot_type in ["normalized", "all"] and "error" in plot_data and "ssim" in plot_data:
        plt.figure(figsize=(10, 5))

        error_data = plot_data["error"]
        ssim_data = plot_data["ssim"]

        # Convert to numpy arrays for calculations
        error_means_np = np.array(error_data["means"])
        error_lower_np = np.array(error_data["lower"])
        error_upper_np = np.array(error_data["upper"])
        ssim_means_np = np.array(ssim_data["means"])
        ssim_lower_np = np.array(ssim_data["lower"])
        ssim_upper_np = np.array(ssim_data["upper"])

        # Determine global min/max for normalization
        error_min_val = np.min(error_lower_np)
        error_max_val = np.max(error_upper_np)
        ssim_min_val = np.min(ssim_lower_np)
        ssim_max_val = np.max(ssim_upper_np)

        error_range = error_max_val - error_min_val if error_max_val > error_min_val else 1.0
        ssim_range = ssim_max_val - ssim_min_val if ssim_max_val > ssim_min_val else 1.0

        # Normalize and invert error (so higher is better)
        norm_error_means = (error_means_np - error_min_val) / error_range
        # For inverted error, swap the CI bounds
        norm_error_lower_ci = (error_upper_np - error_min_val) / error_range
        norm_error_upper_ci = (error_lower_np - error_min_val) / error_range

        # Normalize SSIM (higher is already better)
        norm_ssim_means = (ssim_means_np - ssim_min_val) / ssim_range
        norm_ssim_lower_ci = (ssim_lower_np - ssim_min_val) / ssim_range
        norm_ssim_upper_ci = (ssim_upper_np - ssim_min_val) / ssim_range

        # Plot normalized metrics
        plt.plot(sensor_values, norm_error_means, color="blue", label="Error")
        plt.scatter(sensor_values, norm_error_means, color="blue", marker="o", s=30)
        plt.fill_between(
            sensor_values,
            np.minimum(norm_error_lower_ci, norm_error_upper_ci),
            np.maximum(norm_error_lower_ci, norm_error_upper_ci),
            color="blue",
            alpha=0.2,
        )

        plt.plot(sensor_values, norm_ssim_means, color="green", label="SSIM")
        plt.scatter(sensor_values, norm_ssim_means, color="green", marker="o", s=30)
        plt.fill_between(
            sensor_values, norm_ssim_lower_ci, norm_ssim_upper_ci, color="green", alpha=0.2
        )

        plt.xlabel("Number of Sensors (N)")
        plt.ylabel("Normalized Quality Metrics")
        plt.title("Performance Metrics vs. Number of Sensors")
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1)
        plt.tight_layout()

        if save_path:
            save_plot("normalized")

        if show_plots:
            plt.show()
        else:
            plt.close()

    # Plot 3: Combined Non-Normalized Plot (All metrics, only if explicitly requested)
    if plot_type == "combined":
        plt.figure(figsize=(12, 6))

        for i, (metric, data) in enumerate(plot_data.items()):
            color = colors.get(metric, f"C{i}")
            marker = ["o", "s", "^", "D"][i % 4]  # Different markers

            plt.plot(sensor_values, data["means"], color=color, label=f"Mean {metric.upper()}")
            plt.scatter(sensor_values, data["means"], color=color, marker=marker, s=30)

            # Add confidence intervals
            if not np.array_equal(data["lower"], data["means"]) or not np.array_equal(
                data["upper"], data["means"]
            ):
                plt.fill_between(
                    sensor_values,
                    data["lower"],
                    data["upper"],
                    color=color,
                    alpha=0.2,
                    label=f"{metric.upper()} 95% CI",
                )

        plt.title("Combined Metrics vs. Sensors")
        plt.xlabel("Number of Sensors (N)")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            save_plot("combined")

        if show_plots:
            plt.show()
        else:
            plt.close()


def plot_analytics_legacy(
    sensor_values,
    error_means,
    error_lower,
    error_upper,
    ssim_means,
    ssim_lower,
    ssim_upper,
    psnr_means,
    psnr_lower,
    psnr_upper,
    save_path: Optional[str] = None,
):
    """
    Legacy plot function for backward compatibility.

    This is the original plot_analytics function adapted from plots.py
    for use with separate arrays instead of DataFrame.
    """
    print("Warning: Using legacy plot function. Consider using plot_analytics with DataFrame.")

    # Create a DataFrame and use the new function
    df = pd.DataFrame(
        {
            "sensors": sensor_values,
            "error_mean": error_means,
            "error_ci_lower": error_lower,
            "error_ci_upper": error_upper,
            "ssim_mean": ssim_means,
            "ssim_ci_lower": ssim_lower,
            "ssim_ci_upper": ssim_upper,
            "psnr_mean": psnr_means,
            "psnr_ci_lower": psnr_lower,
            "psnr_ci_upper": psnr_upper,
        }
    )

    plot_analytics(df, metrics=["error", "ssim", "psnr"], plot_type="all", save_path=save_path)
