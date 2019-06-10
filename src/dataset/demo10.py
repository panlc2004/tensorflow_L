import numpy as np

a = [[1, 2], [3, 4]]
b = [5, 6]
c = zip(a, b)
d = list(c)
print(d)
np.random.shuffle(d)
d = np.asarray(d)
print(d)

x = d[:, 0]
y = d[:, 1]
print(x)
print(y)
