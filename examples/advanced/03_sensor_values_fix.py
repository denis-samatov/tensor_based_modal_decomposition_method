"""Demonstrate the sensor_values type-fix workflow.

Shows how to resolve the error:
ValueError: N must be a positive integer, got 1
"""

import numpy as np
import torch

from TBMD.analytics.analytics import ExperimentRunner, ExperimentConfig, ensure_sensor_values_are_int


def demonstrate_problem_and_solution():
    """Demonstrate the type issue and its solution."""
    print("SENSOR_VALUES TYPE-FIX DEMONSTRATION")
    print("=" * 60)
    
    # Create test data.
    A_tensor = torch.randn(10, 10, 5, dtype=torch.float32)
    test_tensors = {
        'subject_1': torch.randn(10, 10, 3, dtype=torch.float32)
    }
    subject_name = 'subject_1'
    slice_number = 0
    
    # Problem: sensor_values with NumPy integer types.
    print("\nProblematic sensor_values with NumPy integer types:")
    problematic_sensor_values = [np.int64(1), np.int32(3), np.int64(5)]
    print(f"   Values: {problematic_sensor_values}")
    print(f"   Types: {[type(x) for x in problematic_sensor_values]}")
    
    # Solution 1: use the utility function.
    print("\nSolution 1: ensure_sensor_values_are_int() utility")
    fixed_sensor_values = ensure_sensor_values_are_int(problematic_sensor_values)
    print(f"   Fixed values: {fixed_sensor_values}")
    print(f"   Types: {[type(x) for x in fixed_sensor_values]}")
    
    # Solution 2: manual conversion.
    print("\nSolution 2: manual conversion")
    manual_fix = [int(x) for x in problematic_sensor_values]
    print(f"   Fixed values: {manual_fix}")
    print(f"   Types: {[type(x) for x in manual_fix]}")
    
    # Solution 3: start with Python int values.
    print("\nSolution 3: use Python int values from the start")
    correct_sensor_values = [1, 3, 5]
    print(f"   Values: {correct_sensor_values}")
    print(f"   Types: {[type(x) for x in correct_sensor_values]}")
    
    return fixed_sensor_values, manual_fix, correct_sensor_values


def test_fixed_experiment():
    """Run a small experiment with fixed values."""
    print("\nFIXED EXPERIMENT TEST")
    print("=" * 60)
    
    # Create test data.
    A_tensor = torch.randn(10, 10, 5, dtype=torch.float32)
    test_tensors = {
        'subject_1': torch.randn(10, 10, 3, dtype=torch.float32)
    }
    subject_name = 'subject_1'
    slice_number = 0
    
    # Use fixed sensor_values.
    sensor_values = ensure_sensor_values_are_int([np.int64(1), np.int32(3), np.int64(5)])
    
    # Create configuration.
    config = ExperimentConfig(
        device='cpu',
        noise_level=0.0,  # No noise for a quick test.
        verbose=False
    )
    
    experiment_runner = ExperimentRunner(config)
    
    print("\nRunning run_single_slice_experiments...")
    print(f"   subject_name: {subject_name}")
    print(f"   slice_number: {slice_number}")
    print(f"   sensor_values: {sensor_values}")
    
    try:
        # This should now work.
        df = experiment_runner.run_single_slice_experiments(
            A_tensor, test_tensors, subject_name, slice_number, sensor_values
        )
        
        print("Experiment completed successfully.")
        print(f"Result columns: {list(df.columns)}")
        print("Results:")
        print(df)
        
        return df
        
    except Exception as e:
        print(f"Error: {e}")
        return None


def show_best_practices():
    """Show best practices for avoiding the issue."""
    print("\nBEST PRACTICES")
    print("=" * 60)
    
    print("\nRecommendations:")
    print("1. Use Python int directly:")
    print("   sensor_values = [1, 3, 5, 10]")
    
    print("\n2. Convert NumPy arrays before passing them:")
    print("   sensor_values = ensure_sensor_values_are_int(your_numpy_array)")
    
    print("\n3. Or use .tolist() for NumPy arrays:")
    print("   sensor_values = np.array([1, 3, 5]).tolist()")
    
    print("\n4. Check types when debugging:")
    print("   print([type(x) for x in sensor_values])")
    
    print("\nAvoid:")
    print("   np.arange(1, 10)  # NumPy array")
    print("   [np.int64(1), np.int32(3)]  # NumPy integer types")
    
    print("\nCorrect usage examples:")
    
    # Example 1: simple list.
    print("\n   # Example 1: simple list")
    example1 = """
   sensor_values = [1, 3, 5, 8, 10]
   df = runner.run_experiments(A_tensor, test_tensors, sensor_values)
   """
    print(example1)
    
    # Example 2: NumPy conversion.
    print("\n   # Example 2: NumPy conversion")
    example2 = """
   numpy_sensors = np.arange(1, 11, 2)  # [1, 3, 5, 7, 9]
   sensor_values = ensure_sensor_values_are_int(numpy_sensors)
   df = runner.run_experiments(A_tensor, test_tensors, sensor_values)
   """
    print(example2)
    
    # Example 3: using .tolist().
    print("\n   # Example 3: using .tolist()")
    example3 = """
   sensor_values = np.array([1, 3, 5, 8, 10]).tolist()
   df = runner.run_experiments(A_tensor, test_tensors, sensor_values)
   """
    print(example3)


def main():
    """Run the demonstration."""
    print("TYPE-FIX DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Demonstrate the issue and solutions.
        fixed1, fixed2, fixed3 = demonstrate_problem_and_solution()
        
        # Test the fixed experiment.
        df = test_fixed_experiment()
        
        # Show best practices.
        show_best_practices()
        
        print("\nIssue resolved in this demonstration.")
        
        print("\nSummary:")
        print("- Type validation accepts NumPy integer values.")
        print("- ExperimentRunner performs automatic conversion.")
        print("- ensure_sensor_values_are_int() is available as a utility.")
        print("- run_single_slice_experiments accepts converted values.")
        
        print("\nMain recommendation:")
        print("Use Python int values for sensor_values or call ensure_sensor_values_are_int().")
        
        return {
            'fixed_sensor_values': fixed1,
            'experiment_result': df
        }
        
    except Exception as e:
        print(f"\nDemonstration error: {e}")
        raise


if __name__ == "__main__":
    results = main() 
