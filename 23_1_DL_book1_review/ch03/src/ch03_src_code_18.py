import numpy as np

a = np.array([1010, 1000, 990])

y = np.exp(a) / np.sum(np.exp(a))

print(y)
'''
[nan nan nan]
'''


c = np.max(a)
a - c
print(a-c)
'''
[  0 -10 -20]
'''


y = np.exp(a-c) / np.sum(np.exp(a-c))

print(y)
'''
[9.99954600e-01 4.53978686e-05 2.06106005e-09]
'''