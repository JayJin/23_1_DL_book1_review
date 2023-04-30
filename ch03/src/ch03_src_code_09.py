import numpy as np

A = np.array([[1, 2], [3, 4]])
print(A.shape)

B = np.array([[5, 6], [7, 8]])
print(B.shape)

print(np.dot(A, B))     # 행렬 곱 : np.dot

# 출력 결과
'''
(2, 2)
(2, 2)
[[19 22]
 [43 50]]
'''
