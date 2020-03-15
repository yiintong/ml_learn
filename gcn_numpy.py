import numpy as np
A = np.matrix([
    [0, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [1, 0, 1, 0]],
    dtype=float)
# 怎么用ndarray生成数组
print(A)
print(A.shape[0])
X = np.matrix([
            [i, -i]
            for i in range(A.shape[0])
        ], dtype=float)
print(X)
