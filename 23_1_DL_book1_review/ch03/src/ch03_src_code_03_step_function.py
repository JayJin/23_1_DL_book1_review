import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x>0, dtype=int)

x = np.arange(-5.0, 5.0, 0.1)       # -5.0 ~ 5.0까지 0.1간격의 배열 생성
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)     # y축의 범위 지정
plt.show()
