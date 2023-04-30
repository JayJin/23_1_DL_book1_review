# softmax 함수의 구현
import numpy as np

a = np.array([0.3, 2.9, 4.0])

exp_a = np.exp(a)
sum_exp_a = np.sum(exp_a)

y = exp_a / sum_exp_a
print(y)


