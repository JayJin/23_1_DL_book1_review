import numpy as np
import matplotlib.pylab as plt

from ch03_src_code_04 import sigmoid


x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()


