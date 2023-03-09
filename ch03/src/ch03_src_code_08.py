import numpy as np

# 1차원 배열 작성
A = np.array([1, 2, 3, 4])
print(A)
print(np.ndim(A))       # 배열 차원
print(A.shape)          # 배열 형상
print(A.shape[0])       # 1차원의 원소 수

# 출력 결과
'''
[1 2 3 4]
1           # 1차원
(4,)        # 4x1 배열(열, 행)
4
'''

# 2차원 배열 작성
B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
print(np.ndim(B))
print(B.shape)

# 출력 결과
'''
[[1 2]
 [3 4]
 [5 6]]
2             # 2차원      
(3, 2)        # 3x2 배열(열, 행)
'''