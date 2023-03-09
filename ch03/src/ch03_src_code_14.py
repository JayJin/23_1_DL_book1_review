import numpy as np

from ch03_src_code_04 import sigmoid
from ch03_src_code_13 import Z2

def identity_function(x):       # 항등함수 정의
    return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)

print(A3)
print(Y)        # A = Y3

# 출력 결과
'''
[0.31682708 0.69627909]
[0.31682708 0.69627909]
'''