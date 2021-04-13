import numpy as np
a = np.array([[1, 2], [3, 4], [5, 6]])
b = np.array([[7, 8]])
a = np.append(a, b, axis=0)
a = a.T
print(a)