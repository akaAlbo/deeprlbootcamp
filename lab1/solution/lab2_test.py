from __future__ import print_function

import numpy as np

v = np.arange(0, 16).reshape(-1, 1)
# alpha = np.array([[1.0,] *16, ] * 16)
alpha = np.arange(0, 16*16).reshape(16, 16)
beta = 2.5
A = np.array([[2.0, ] * 16, ] * 16)


# print(v.shape)  # (16, 16)
# print(alpha.shape)  # (16, 1)
# print(A.shape)  # (16, 16)



a = np.array([[3,1], [1,2]])
print(a)
b = np.array([[9],[8]])
print(b)
x = np.linalg.solve(a, b)
print(x)