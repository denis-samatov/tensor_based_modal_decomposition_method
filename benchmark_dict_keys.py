import timeit
import time
import os

setup_code = """
d = {f'key_{i}': i for i in range(10000)}
"""

test_list = """
k = list(d)
"""

test_iter = """
k = d.keys()
"""

print("Benchmarking list(d) vs d.keys()")
time_list = timeit.timeit(test_list, setup=setup_code, number=100000)
time_iter = timeit.timeit(test_iter, setup=setup_code, number=100000)

print(f"list(d): {time_list:.6f} seconds")
print(f"d.keys(): {time_iter:.6f} seconds")
print(f"Speedup: {time_list / time_iter:.2f}x")
