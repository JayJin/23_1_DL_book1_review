import sys, os
# sys.path.append(os.pardir)
sys.path.append(os.getcwd())

import numpy as np
from dataset.mnist import load_mnist

# 훈련 데이터, 테스트 데이터 분할
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
print(x_train.shape)        # (60000, 784)
print(t_train.shape)        # (60000, 10)

# 훈련데이터에서 무작위로 10개 추출
train_size = x_train.shape[0]       # 범위
batch_size = 10                     # 원하는 갯수
batch_mask = np.random.choice(train_size, batch_size)       # 지정한 범위내에서 무작위로 원하는 갯수만큼 추출
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

# 교차 엔트로피 오차 함수구현
def cross_entropy_error(y, t):
    # y가 1차원일 경우 reshape를 통해 데이터 형상을 바꾸어준다.
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]

    # 결과 리턴 (1) _ 배치 사이즈로 나누어 정규화(1장당 평균) - one-hot-encoding일 경우
    return -np.sum(t * np.log(y)) / batch_size

    # 결과 리턴 (2)_ 배치 사이즈로 나누어 정규화(1장당 평균) - one-hot-encoding이 아닐 경우
    # return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size      

