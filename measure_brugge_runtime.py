import h5py
import numpy as np
import scipy.linalg
import time
import sys
import os
import resource

sys.path.insert(0, os.path.abspath('src'))

from TBMD.core.decomposition.hosvd import TuckerDecomposerInterface
from TBMD.config import DecompositionConfig

def format_bytes(b):
    if b < 1024:
        return f"{b} B"
    elif b < 1024**2:
        return f"{b / 1024:.2f} KB"
    elif b < 1024**3:
        return f"{b / 1024**2:.2f} MB"
    else:
        return f"{b / 1024**3:.2f} GB"

def get_memory_usage():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == 'darwin':
        return usage
    else:
        return usage * 1024

def main():
    h5_path = 'data/brugge/data_exp_4_.h5'
    print(f"Loading data from {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        pressure = f['pressure'][0]  
        saturation = f['soil'][0]    
    
    tensor = np.stack([pressure, saturation], axis=2)
    print(f"Constructed tensor shape: {tensor.shape}")
    
    # 1. TBMD Runtime & Memory
    mem_before_tbmd = get_memory_usage()
    start_time = time.time()
    
    config = DecompositionConfig(
        method="hosvd",
        epsilon=1e-2,
    )
    decomposer = TuckerDecomposerInterface(tensor, config=config)
    decomposer.decompose()
    
    tbmd_time = time.time() - start_time
    mem_after_tbmd = get_memory_usage()
    
    tbmd_peak = mem_after_tbmd - mem_before_tbmd
    print(f"TBMD Wall Time: {tbmd_time:.3f} s")
    print(f"TBMD Peak Memory (RSS delta): {format_bytes(tbmd_peak)} (Absolute peak: {format_bytes(mem_after_tbmd)})")
    
    # 2. QR Runtime & Memory
    factors = decomposer.factors
    U1, U2, U3 = factors[0], factors[1], factors[2]
    rank = U1.shape[1] * U2.shape[1] * U3.shape[1]
    n_space = U1.shape[0] * U2.shape[0] * U3.shape[0]
    print(f"QR dictionary shape: {(n_space, rank)}")
    
    dummy_basis = np.random.randn(n_space, min(rank, n_space))
    
    mem_before_qr = get_memory_usage()
    start_qr = time.time()
    
    q, r, p = scipy.linalg.qr(dummy_basis, mode='economic', pivoting=True)
    
    qr_time = time.time() - start_qr
    mem_after_qr = get_memory_usage()
    qr_peak = mem_after_qr - mem_before_qr
    
    print(f"QR Wall Time: {qr_time:.3f} s")
    print(f"QR Peak Memory (RSS delta): {format_bytes(qr_peak)} (Absolute peak: {format_bytes(mem_after_qr)})")
    
    # 3. CS Recovery (ADMM)
    from TBMD.core.reconstruction.tensor_compressive_sensing import TensorCompressiveSensing, CompressiveSensingConfig
    
    # A is (n_space, rank) dictionary, but flattened. Here dummy_basis is exactly that.
    A = dummy_basis
    # We select e.g., 20 sensors randomly
    P = np.zeros(n_space, dtype=bool)
    P[np.random.choice(n_space, 20, replace=False)] = True
    # Measurements Y correspond to the active sensors. Y shape is (n_space, ) but only P=True are used
    Y = np.random.randn(n_space)
    
    mem_before_cs = get_memory_usage()
    start_cs = time.time()
    
    cs_cfg = CompressiveSensingConfig(max_iter=1000, tol=1e-4)
    cs_solver = TensorCompressiveSensing(A, P, Y, core_cfg=cs_cfg)
    solution, metrics = cs_solver.solve()
    
    cs_time = time.time() - start_cs
    mem_after_cs = get_memory_usage()
    cs_peak = mem_after_cs - mem_before_cs
    
    print(f"CS Recovery Wall Time: {cs_time:.3f} s")
    print(f"CS Peak Memory (RSS delta): {format_bytes(cs_peak)} (Absolute peak: {format_bytes(mem_after_cs)})")
    
if __name__ == '__main__':
    main()
