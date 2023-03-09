# 시그모이드 함수와 계단함수 비교하는 plot 그리기
import numpy as np
import matplotlib.pylab as plt

from ch03_src_code_03 import step_function
from ch03_src_code_04 import sigmoid

x = np.arange(-5.0, 5.0, 0.1)
y1 = step_function(x)
y2 = sigmoid(x)
plt.plot(x, y1, linestyle="--", label='step')
plt.plot(x, y2)
plt.ylim(-0.1, 1.1)
plt.show()
