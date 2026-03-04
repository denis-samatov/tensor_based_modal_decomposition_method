import timeit

setup_code_append = """
dim = 3
size = 256
"""

stmt_append = """
size_list = []
for j in range(dim):
    size_list.append(size)
tuple(size_list)
"""

stmt_list_comp = """
size_list = [size for _ in range(dim)]
tuple(size_list)
"""

print(f"Append baseline: {timeit.timeit(stmt=stmt_append, setup=setup_code_append, number=1000000)} seconds")
print(f"List comprehension: {timeit.timeit(stmt=stmt_list_comp, setup=setup_code_append, number=1000000)} seconds")
