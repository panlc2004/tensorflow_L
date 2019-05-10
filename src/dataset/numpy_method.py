import numpy as np
import pandas as pd

dataset = np.loadtxt("test.csv", delimiter=",", skiprows=2)
print(dataset)

X = dataset[:, 0:1]
Y = dataset[:, 1]
print(X)
print(Y)