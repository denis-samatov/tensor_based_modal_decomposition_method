SEED = 42

##########################################################################
# Tensor-based Tube Fiber-Pivot QR Factorization (TTFP-QR) Configuration
##########################################################################

NUMBER_SENSORS = 200          # Number of physical/virtual sensors in the array 
                              # (defines 3rd-order tensor dimension: I × J × K)

#######################################################################
# Tensor-based Compressive Sensing Algorithm (TCSA) Parameters
#######################################################################

# Optimization settings
MAX_ITERATIONS = 100          # Maximum optimization iterations
CONVERGENCE_EPS = 1e-2        # Convergence threshold (||X_{k+1} - X_k|| < eps)
DAMPING_FACTOR = 0.95         # Regularization parameter (λ) for tensor decomposition
INITIAL_STEP_SIZE = 1.0       # Starting value for adaptive step size (Δ₀)
MAX_STEP_SIZE = 1.0           # Upper bound for step size adjustment (Δ_max)


SET_BACKEND = 'pytorch'
DTYPE = 'float32'