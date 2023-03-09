import numpy as np

def step_function(x):
    y = x>0
    return y.astype(int)

x = np.array([-1.0, 1.0, 2.0])
print(x)

y = x > 0
print(y)        # y값이 bool로 표현

y = y.astype(int)
print(y)        # bool로 표현되던 y값이 0, 1로 변환됨

# 출력 결과
'''
[-1.  1.  2.]
[False  True  True]
[0 1 1]
'''

# 주의. y.astype(np.int)로 사용시 아래 오류 발생
# AttributeError: module 'numpy' has no attribute 'int'. 

# numpy This happens because you have installed the latest numpy version where np.int is depreciated.
# Instead install a lower version of numpy pip install "numpy<1.24.0".
# Or you can change the np.int to just int.