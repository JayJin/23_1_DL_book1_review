import numpy as np

X = np.array([1, 2])
print(X.shape)

W = np.array([[1, 3, 5], [2, 4, 6]])
print(W)
print(W.shape)

Y = np.dot(X, W)
print(Y)

# 출력 결과
'''
(2,)
[[1 3 5]
 [2 4 6]]
(2, 3)
[ 5 11 17]
'''