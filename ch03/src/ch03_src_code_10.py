import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
print(A.shape)

B = np.array([[1, 2], [3, 4], [5, 6]])
print(B.shape)

print(np.dot(A,B))

# 출력 결과
'''
(2, 3)
(3, 2)
[[22 28]
 [49 64]]
'''