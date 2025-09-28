#!/usr/bin/env python3
"""
Example usage of the improved TensorHOSVD module.

This script demonstrates the new architecture with proper error handling,
state management, and improved API.
"""

import torch
import numpy as np
from TBMD.modules.TensorHOSVD import (
    TuckerDecomposerInterface,
    IterativeHOSVDDecomposer,
    TensorDecompositionError,
    InvalidRankError,
    StateError,
    ValidationError
)

def create_sample_data():
    """Create sample tensor data for demonstration"""
    print("Creating sample tensor data...")
    
    # Single tensor
    single_tensor = torch.randn(50, 40, 30, dtype=torch.float32)
    
    # Collection of tensors
    tensor_collection = {
        'subject_1': torch.randn(50, 40, 30, dtype=torch.float32),
        'subject_2': torch.randn(50, 40, 30, dtype=torch.float32),
        'subject_3': torch.randn(50, 40, 30, dtype=torch.float32),
    }
    
    return single_tensor, tensor_collection

def demonstrate_single_tensor_decomposition():
    """Demonstrate decomposition of a single tensor"""
    print("\n" + "="*50)
    print("SINGLE TENSOR DECOMPOSITION DEMO")
    print("="*50)
    
    single_tensor, _ = create_sample_data()
    
    try:
        # Initialize with improved interface
        decomposer = TuckerDecomposerInterface(
            tensors=single_tensor,
            ranks=[20, 15, 10],  # Custom ranks per mode
            epsilon=1e-3,
            device='cpu',
            random_state=42
        )
        
        print(f"✓ Initialized decomposer")
        print(f"  Tensor shape: {single_tensor.shape}")
        print(f"  Device: {decomposer.processor.device}")
        print(f"  State: {decomposer.state.value}")
        
        # Perform decomposition
        decomposer.decompose()
        print(f"✓ Decomposition completed")
        print(f"  State: {decomposer.state.value}")
        print(f"  Core shape: {decomposer.cores.shape}")
        print(f"  Number of factors: {len(decomposer.factors)}")
        
        # Reconstruct
        decomposer.reconstruct()
        print(f"✓ Reconstruction completed")
        print(f"  State: {decomposer.state.value}")
        print(f"  Reconstruction error: {decomposer.reconstruction_errors:.6f}")
        
        # Demonstrate property access
        cores = decomposer.cores
        factors = decomposer.factors
        reconstructed = decomposer.reconstructed_tensors
        
        print(f"✓ Successfully accessed all results")
        
    except (TensorDecompositionError, ValidationError, StateError) as e:
        print(f"✗ Error: {e}")
        return False
    
    return True

def demonstrate_collection_decomposition():
    """Demonstrate decomposition of tensor collection"""
    print("\n" + "="*50)
    print("TENSOR COLLECTION DECOMPOSITION DEMO")
    print("="*50)
    
    _, tensor_collection = create_sample_data()
    
    try:
        # Initialize for collection processing
        decomposer = TuckerDecomposerInterface(
            tensors=tensor_collection,
            ranks=15,  # Uniform rank for all modes
            epsilon=1e-3,
            device='cpu',
            max_workers=2,  # Limit parallel workers
            random_state=42
        )
        
        print(f"✓ Initialized collection decomposer")
        print(f"  Number of tensors: {len(tensor_collection)}")
        print(f"  Processing strategy: {type(decomposer.strategy).__name__}")
        
        # Perform decomposition
        decomposer.decompose()
        print(f"✓ Collection decomposition completed")
        
        # Show cores info
        cores = decomposer.cores
        for key, core in cores.items():
            print(f"  {key}: core shape {core.shape}")
        
        # Reconstruct
        decomposer.reconstruct()
        print(f"✓ Collection reconstruction completed")
        
        # Show reconstruction errors
        errors = decomposer.reconstruction_errors
        for key, error in errors.items():
            print(f"  {key}: error {error:.6f}")
        
        # Visualize (comment out if running headless)
        # decomposer.visualize(subjects=['subject_1'])
        
    except (TensorDecompositionError, ValidationError, StateError) as e:
        print(f"✗ Error: {e}")
        return False
    
    return True

def demonstrate_error_handling():
    """Demonstrate improved error handling"""
    print("\n" + "="*50)
    print("ERROR HANDLING DEMO")
    print("="*50)
    
    single_tensor, _ = create_sample_data()
    
    # Test validation errors
    print("Testing input validation...")
    
    try:
        # Invalid ranks
        TuckerDecomposerInterface(single_tensor, ranks=[-1, 5, 10])
        print("✗ Should have failed with negative rank")
    except InvalidRankError as e:
        print(f"✓ Caught expected error: {e}")
    
    try:
        # Ranks too large
        TuckerDecomposerInterface(single_tensor, ranks=[100, 5, 10])
        print("✗ Should have failed with rank too large")
    except InvalidRankError as e:
        print(f"✓ Caught expected error: {e}")
    
    try:
        # Invalid epsilon
        TuckerDecomposerInterface(single_tensor, epsilon=-0.1)
        print("✗ Should have failed with negative epsilon")
    except ValidationError as e:
        print(f"✓ Caught expected error: {e}")
    
    # Test state management
    print("\nTesting state management...")
    
    decomposer = TuckerDecomposerInterface(single_tensor, ranks=10)
    
    try:
        # Try to reconstruct before decomposing
        decomposer.reconstruct()
        print("✗ Should have failed - wrong state")
    except StateError as e:
        print(f"✓ Caught expected state error: {e}")
    
    try:
        # Try to access results before decomposing
        _ = decomposer.cores
        print("✗ Should have failed - no cores yet")
    except StateError as e:
        print(f"✓ Caught expected state error: {e}")
    
    # Proper workflow
    decomposer.decompose()
    
    try:
        # Try to change ranks after decomposition
        decomposer.set_ranks(5)
        print("✗ Should have failed - can't change ranks after decomposition")
    except StateError as e:
        print(f"✓ Caught expected state error: {e}")

def demonstrate_performance_comparison():
    """Demonstrate performance improvements"""
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON DEMO")
    print("="*50)
    
    import time
    
    # Create larger dataset for timing
    large_collection = {
        f'tensor_{i}': torch.randn(30, 25, 20, dtype=torch.float32)
        for i in range(6)
    }
    
    print(f"Testing with {len(large_collection)} tensors...")
    
    # Test with different worker counts
    for max_workers in [1, 2, 4]:
        start_time = time.time()
        
        decomposer = TuckerDecomposerInterface(
            tensors=large_collection,
            ranks=10,
            max_workers=max_workers,
            device='cpu'
        )
        
        decomposer.decompose()
        decomposer.reconstruct()
        
        elapsed = time.time() - start_time
        print(f"  {max_workers} workers: {elapsed:.2f} seconds")

def demonstrate_iterative_hosvd():
    """Demonstrate the true iterative HOSVD algorithm from Algorithm 1"""
    print("\n" + "="*50)
    print("ITERATIVE HOSVD ALGORITHM DEMO")
    print("="*50)
    
    # Create a 3D tensor for HOSVD
    tensor_3d = torch.randn(20, 15, 10, dtype=torch.float32)
    print(f"Created 3D tensor with shape: {tensor_3d.shape}")
    
    try:
        # Initialize iterative HOSVD decomposer
        hosvd = IterativeHOSVDDecomposer(
            epsilon=1e-3,
            max_iterations=50,
            rank_reduction_strategy='decrement',  # or 'fixed'
            random_state=42
        )
        
        print(f"✓ Initialized iterative HOSVD decomposer")
        print(f"  Epsilon: {hosvd.epsilon}")
        print(f"  Max iterations: {hosvd.max_iterations}")
        print(f"  Rank strategy: {hosvd.rank_reduction_strategy}")
        
        # Perform iterative HOSVD decomposition
        print("\nPerforming iterative HOSVD decomposition...")
        core, factors = hosvd.decompose_iterative_hosvd(
            tensor_3d, 
            initial_ranks=[10, 8, 6]  # Custom initial ranks
        )
        
        print(f"✓ Iterative HOSVD completed")
        print(f"  Core tensor shape: {core.shape}")
        print(f"  Factor matrices shapes: {[f.shape for f in factors]}")
        
        # Get convergence information
        conv_info = hosvd.get_convergence_info()
        print(f"  Converged: {conv_info['converged']}")
        print(f"  Iterations used: {conv_info['iterations_used']}")
        print(f"  Final residual: {conv_info['final_residual']:.6f}")
        
        # Show convergence history
        history = conv_info['convergence_history']
        if len(history) >= 5:
            print("\nConvergence history (first 5 iterations):")
            for i in range(min(5, len(history))):
                iter_info = history[i]
                print(f"  Iter {iter_info['iteration']}: "
                      f"residual = {iter_info['absolute_residual']:.6f}, "
                      f"ranks = {iter_info['ranks']}")
        
        # Compare with standard Tucker decomposition
        print("\nComparing with standard Tucker decomposition...")
        
        # Standard approach
        standard_decomposer = TuckerDecomposerInterface(
            tensors=tensor_3d,
            ranks=[6, 6, 6],  # Fixed ranks similar to final HOSVD ranks
            epsilon=1e-3,
            random_state=42
        )
        standard_decomposer.decompose()
        standard_decomposer.reconstruct()
        
        print(f"✓ Standard Tucker completed")
        print(f"  Standard reconstruction error: {standard_decomposer.reconstruction_errors:.6f}")
        
        # Reconstruct from HOSVD result for comparison
        hosvd_reconstructed = core.clone()
        for n in range(3):
            # Multiply back with factor matrices
            hosvd_reconstructed = torch.tensordot(
                hosvd_reconstructed, factors[n], dims=([n], [1])
            )
            # Rearrange dimensions back
            dims = list(range(hosvd_reconstructed.ndim))
            dims[n], dims[-1] = dims[-1], dims[n]
            hosvd_reconstructed = hosvd_reconstructed.permute(dims)
        
        hosvd_error = torch.norm(tensor_3d - hosvd_reconstructed) / torch.norm(tensor_3d)
        print(f"  HOSVD reconstruction error: {hosvd_error:.6f}")
        
        return True
        
    except (TensorDecompositionError, ValidationError, StateError) as e:
        print(f"✗ Error in iterative HOSVD: {e}")
        return False

def demonstrate_algorithm_comparison():
    """Compare different decomposition algorithms"""
    print("\n" + "="*50)
    print("ALGORITHM COMPARISON DEMO")
    print("="*50)
    
    # Create test tensor
    tensor_3d = torch.randn(15, 12, 8, dtype=torch.float32)
    print(f"Test tensor shape: {tensor_3d.shape}")
    
    import time
    
    algorithms = [
        ("Standard Tucker", lambda: TuckerDecomposerInterface(
            tensors=tensor_3d, ranks=[8, 8, 6], epsilon=1e-3, random_state=42
        )),
        ("Iterative HOSVD (decrement)", lambda: IterativeHOSVDDecomposer(
            epsilon=1e-3, rank_reduction_strategy='decrement', random_state=42
        )),
        ("Iterative HOSVD (fixed)", lambda: IterativeHOSVDDecomposer(
            epsilon=1e-3, rank_reduction_strategy='fixed', random_state=42
        ))
    ]
    
    results = {}
    
    for name, algo_factory in algorithms:
        print(f"\nTesting {name}...")
        
        try:
            start_time = time.time()
            
            if "Tucker" in name:
                decomposer = algo_factory()
                decomposer.decompose()
                decomposer.reconstruct()
                error = decomposer.reconstruction_errors
                iterations = 1  # Single shot
                final_ranks = decomposer.decomposer.ranks
            else:
                decomposer = algo_factory()
                core, factors = decomposer.decompose_iterative_hosvd(tensor_3d)
                conv_info = decomposer.get_convergence_info()
                
                # Compute reconstruction error manually
                reconstructed = core.clone()
                for n in range(3):
                    reconstructed = torch.tensordot(
                        reconstructed, factors[n], dims=([n], [1])
                    )
                    dims = list(range(reconstructed.ndim))
                    dims[n], dims[-1] = dims[-1], dims[n]
                    reconstructed = reconstructed.permute(dims)
                
                error = float(torch.norm(tensor_3d - reconstructed) / torch.norm(tensor_3d))
                iterations = conv_info['iterations_used']
                final_ranks = [f.shape[1] for f in factors]
            
            elapsed = time.time() - start_time
            
            results[name] = {
                'error': error,
                'time': elapsed,
                'iterations': iterations,
                'final_ranks': final_ranks
            }
            
            print(f"  ✓ Completed in {elapsed:.3f}s")
            print(f"    Reconstruction error: {error:.6f}")
            print(f"    Iterations: {iterations}")
            print(f"    Final ranks: {final_ranks}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[name] = {'error': float('inf'), 'time': float('inf')}
    
    # Summary comparison
    print("\n" + "="*30)
    print("ALGORITHM COMPARISON SUMMARY")
    print("="*30)
    
    for name, result in results.items():
        if result['error'] != float('inf'):
            print(f"{name}:")
            print(f"  Error: {result['error']:.6f}")
            print(f"  Time: {result['time']:.3f}s")
            print(f"  Iterations: {result['iterations']}")
            print(f"  Final ranks: {result['final_ranks']}")
        else:
            print(f"{name}: FAILED")

def main():
    """Run all demonstrations"""
    print("IMPROVED TENSOR HOSVD DEMONSTRATION")
    print("="*60)
    
    success_count = 0
    total_tests = 5  # Updated count
    
    # Run demonstrations
    if demonstrate_single_tensor_decomposition():
        success_count += 1
    
    if demonstrate_collection_decomposition():
        success_count += 1
    
    if demonstrate_error_handling():
        success_count += 1
    
    if demonstrate_iterative_hosvd():
        success_count += 1
    
    demonstrate_performance_comparison()
    demonstrate_algorithm_comparison()
    
    print("\n" + "="*60)
    print(f"SUMMARY: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("✓ All improvements working correctly!")
        print("✓ Both standard Tucker and iterative HOSVD algorithms available!")
    else:
        print("✗ Some issues detected")

if __name__ == "__main__":
    main() 