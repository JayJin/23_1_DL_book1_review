import numpy as np

x = np.array([0, 1])
w = np.array([0.5, 0.5])
b = -0.7

print(w*x)
print(np.sum(w*x))
print(np.sum(w*x)+b)

# 출력결과
'''
[0.  0.5]
0.5
-0.19999999999999996   # 부동소수점 수에 의한 연산 오차 발생
'''