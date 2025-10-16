"""The global random seed for reproducibility."""
SEED = 42

##########################################################################
# Tensor-based Tube Fiber-Pivot QR Factorization (TTFP-QR) Configuration
##########################################################################

"""The number of physical/virtual sensors in the array.

This defines the 3rd-order tensor dimension: I × J × K.
"""
NUMBER_SENSORS = 200

#######################################################################
# Tensor-based Compressive Sensing Algorithm (TCSA) Parameters
#######################################################################

# Optimization settings
"""The maximum number of optimization iterations."""
MAX_ITERATIONS = 100
"""The convergence threshold (||X_{k+1} - X_k|| < eps)."""
CONVERGENCE_EPS = 1e-2
"""The regularization parameter (λ) for tensor decomposition."""
DAMPING_FACTOR = 0.95
"""The starting value for the adaptive step size (Δ₀)."""
INITIAL_STEP_SIZE = 1.0
"""The upper bound for step size adjustment (Δ_max)."""
MAX_STEP_SIZE = 1.0


"""The backend to use for tensor operations."""
SET_BACKEND = 'pytorch'
"""The data type to use for tensors."""
DTYPE = 'float32'