"""Demonstration of the unified run_experiments method.

Shows automatic analysis-mode selection and several usage patterns.
"""

import numpy as np
import torch

from TBMD.analytics.analytics import ExperimentRunner, ExperimentConfig, plot_analytics


def create_sample_data():
    """Create sample data for the demonstration."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    A_tensor = torch.randn(32, 32, 10, dtype=torch.float32)
    test_tensors = {
        'subject_1': torch.randn(32, 32, 5, dtype=torch.float32),
        'subject_2': torch.randn(32, 32, 5, dtype=torch.float32)
    }
    sensor_values = [5, 10, 15, 20]
    
    return A_tensor, test_tensors, sensor_values


def demo_auto_detection():
    """Demonstrate automatic analysis-mode detection."""
    print("AUTOMATIC MODE DETECTION")
    print("=" * 60)
    
    A_tensor, test_tensors, sensor_values = create_sample_data()
    
    # Configuration 1: no noise -> simple mode.
    print("\nConfiguration 1: no noise")
    config1 = ExperimentConfig(device='cpu', noise_level=0.0, num_noise_samples=0)
    runner1 = ExperimentRunner(config1)
    
    df1 = runner1.run_experiments(A_tensor, test_tensors, sensor_values)
    print("Automatically selected simple mode")
    print(f"Columns: {list(df1.columns)}")
    print(f"Data:\n{df1.head()}")
    
    # Configuration 2: noise -> statistical mode.
    print("\nConfiguration 2: noise-aware sampling")
    config2 = ExperimentConfig(device='cpu', noise_level=0.1, num_noise_samples=3, 
                             noise_threshold=1e-10)
    runner2 = ExperimentRunner(config2)
    
    df2 = runner2.run_experiments(A_tensor, test_tensors, sensor_values)
    print("Automatically selected statistical mode")
    print(f"Columns: {list(df2.columns)}")
    print(f"Data sample:\n{df2[['sensors', 'error_mean', 'error_std']].head()}")
    
    return df1, df2


def demo_explicit_control():
    """Demonstrate explicit analysis-mode control."""
    print("\nEXPLICIT MODE CONTROL")
    print("=" * 60)
    
    A_tensor, test_tensors, sensor_values = create_sample_data()
    
    # Same runner, different modes.
    config = ExperimentConfig(device='cpu', noise_level=0.1, num_noise_samples=5)
    runner = ExperimentRunner(config)
    
    # Force simple mode.
    print("\nForce simple mode (statistical_analysis=False):")
    df_simple = runner.run_experiments(A_tensor, test_tensors, sensor_values, 
                                     statistical_analysis=False)
    print(f"Columns: {list(df_simple.columns)}")
    
    # Force statistical mode.
    print("\nForce statistical mode (statistical_analysis=True):")
    df_stat = runner.run_experiments(A_tensor, test_tensors, sensor_values, 
                                   statistical_analysis=True)
    print(f"Columns: {list(df_stat.columns)}")
    
    # Automatic mode uses config.
    print("\nAutomatic mode (statistical_analysis=None):")
    df_auto = runner.run_experiments(A_tensor, test_tensors, sensor_values)
    print(f"Columns: {list(df_auto.columns)}")
    print("Automatically selected statistical mode because noise_level > 0")
    
    return df_simple, df_stat, df_auto


def demo_backward_compatibility():
    """Demonstrate backward compatibility."""
    print("\nBACKWARD COMPATIBILITY")
    print("=" * 60)
    
    A_tensor, test_tensors, sensor_values = create_sample_data()
    
    config = ExperimentConfig(device='cpu')
    runner = ExperimentRunner(config)
    
    print("\nLegacy run_standard_experiments method:")
    df_old = runner.run_standard_experiments(A_tensor, test_tensors, sensor_values)
    print(f"Columns: {list(df_old.columns)}")
    
    print("\nNew run_experiments method (simple):")
    df_new = runner.run_experiments(A_tensor, test_tensors, sensor_values, 
                                  statistical_analysis=False)
    print(f"Columns: {list(df_new.columns)}")
    
    # Check that results are identical.
    are_equal = df_old.equals(df_new)
    print(f"\nResults are identical: {are_equal}")
    
    return df_old, df_new


def demo_performance_comparison():
    """Compare performance across modes."""
    print("\nPERFORMANCE COMPARISON")
    print("=" * 60)
    
    import time
    
    A_tensor, test_tensors, sensor_values = create_sample_data()
    config = ExperimentConfig(device='cpu', noise_level=0.1, num_noise_samples=10)
    runner = ExperimentRunner(config)
    
    # Measure simple mode.
    start_time = time.time()
    df_simple = runner.run_experiments(A_tensor, test_tensors, sensor_values[:2], 
                                     statistical_analysis=False)
    simple_time = time.time() - start_time
    
    # Measure statistical mode.
    start_time = time.time()
    df_stat = runner.run_experiments(A_tensor, test_tensors, sensor_values[:2], 
                                   statistical_analysis=True)
    stat_time = time.time() - start_time
    
    print(f"Simple mode: {simple_time:.2f} seconds")
    print(f"Statistical mode: {stat_time:.2f} seconds")
    print(f"Slowdown: {stat_time/simple_time:.1f}x")
    
    print(f"\nSimple mode columns: {list(df_simple.columns)}")
    print(f"Statistical mode columns: {list(df_stat.columns)}")
    
    return df_simple, df_stat


def demo_plotting_compatibility():
    """Demonstrate compatibility with plot_analytics."""
    print("\nPLOT_ANALYTICS COMPATIBILITY")
    print("=" * 60)
    
    A_tensor, test_tensors, sensor_values = create_sample_data()
    
    config1 = ExperimentConfig(device='cpu', noise_level=0.0)
    config2 = ExperimentConfig(device='cpu', noise_level=0.1, num_noise_samples=3)
    
    runner1 = ExperimentRunner(config1)
    runner2 = ExperimentRunner(config2)
    
    # Produce data in different formats.
    df_simple = runner1.run_experiments(A_tensor, test_tensors, sensor_values)
    df_stat = runner2.run_experiments(A_tensor, test_tensors, sensor_values)
    
    print("\nSimple data is compatible with plot_analytics:")
    plot_analytics(df_simple, plot_type='combined', title_prefix="Simple Mode", show_plots=False)
    print("Plot created successfully")
    
    print("\nStatistical data is compatible with plot_analytics:")
    plot_analytics(df_stat, plot_type='normalized', title_prefix="Statistical Mode", show_plots=False)
    print("Plot with confidence intervals created successfully")
    
    return df_simple, df_stat


def main():
    """Run the demonstration."""
    print("UNIFIED run_experiments DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Demonstrate all capabilities.
        df1, df2 = demo_auto_detection()
        df3, df4, df5 = demo_explicit_control()
        df6, df7 = demo_backward_compatibility()
        df8, df9 = demo_performance_comparison()
        df10, df11 = demo_plotting_compatibility()
        
        print("\nAll demonstrations completed successfully.")
        
        print("\nSummary:")
        print("- One run_experiments method replaces two legacy entry points.")
        print("- Mode selection can be automatic based on configuration.")
        print("- statistical_analysis provides explicit mode control.")
        print("- Backward compatibility is preserved.")
        print("- plot_analytics accepts both simple and statistical outputs.")
        print("- Simple mode is faster; statistical mode provides uncertainty statistics.")
        print("- Noise-aware sampling preserves the meaning of zero-valued regions.")
        
        print("\nUsage recommendations:")
        print("- Use run_experiments() for new experiments.")
        print("- Configure noise_level and num_noise_samples for automatic mode selection.")
        print("- Use statistical_analysis=False for quick analysis.")
        print("- Use statistical_analysis=True for uncertainty-aware analysis.")
        print("- run_standard_experiments() remains available for backward compatibility.")
        print("- For reservoir data, use noise_threshold for noise-aware sampling.")
        
        return {
            'auto_simple': df1, 'auto_stat': df2,
            'explicit_simple': df3, 'explicit_stat': df4, 'explicit_auto': df5,
            'backward_old': df6, 'backward_new': df7,
            'perf_simple': df8, 'perf_stat': df9,
            'plot_simple': df10, 'plot_stat': df11
        }
        
    except Exception as e:
        print(f"\nDemonstration error: {e}")
        raise


if __name__ == "__main__":
    results = main() 
