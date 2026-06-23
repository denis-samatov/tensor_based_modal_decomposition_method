# Exceptions and Errors

## Purpose
Documents common errors encountered when interfacing with the core library.

## Audience
Developers troubleshooting their scripts.

## Details

### 1. Shape Mismatch Errors
- **Symptom**: `RuntimeError: The size of tensor a (X) must match the size of tensor b (Y) at non-singleton dimension N`
- **Cause**: The input tensor shape does not align with the rank configuration provided in `DecompositionConfig`, or the features dimension is misaligned.
- **Resolution**: Verify the input data shape.

### 2. Ill-Conditioned Matrix Inversions
- **Symptom**: `torch.linalg.LinAlgError: Matrix is not invertible`
- **Cause**: This can occur during the ADMM reconstruction or pseudo-inverse steps if the selected tensor ranks are too large, leading to rank deficiency.
- **Resolution**: Reduce the truncation ranks in the configuration or add regularization.

### 3. Out of Memory (OOM)
- **Symptom**: `CUDA out of memory` or script killed by the OS.
- **Cause**: Core tensor contractions (e.g., in HOSVD) require significant memory. Unfolding large high-dimensional tensors can easily exceed VRAM/RAM limits.
- **Resolution**: Use `BatchModalProcessor` if available, or downsample the spatial grid before decomposition.
