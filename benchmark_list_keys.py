import timeit

setup = "d = {i: i for i in range(1000000)}"

stmt1 = "list(d.keys())"
stmt2 = "list(d)"

t1 = timeit.timeit(stmt1, setup=setup, number=100)
t2 = timeit.timeit(stmt2, setup=setup, number=100)

print(f"list(d.keys()): {t1:.4f} s")
print(f"list(d):        {t2:.4f} s")
