import timeit


def func():
    print(1)


t = timeit.timeit(func, number=10)
print(t)