import numpy as np

from ch03_src_code_04 import sigmoid

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1

print(A1)

# 출력 결과
'''
(2, 3)
(2,)
(3,)
[0.3 0.7 1.1]
'''

Z1 = sigmoid(A1)

print(Z1)

# 출력 결과
'''
[0.3 0.7 1.1]
[0.57444252 0.66818777 0.75026011]
'''