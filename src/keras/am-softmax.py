import numpy as np

x = np.arange(12).reshape((3, 4))
w = np.arange(16).reshape((4, 4)) - 8
y = np.array([0, 3, 1]).reshape((-1, 1))
s = 30
m = 0.35
print('x= \n', x)
print('w= \n', w)
print('y =', y, ', s = ', s, ', m = ', m)

x_l2 = (np.sum(x ** 2, axis=1) ** (1 / 2)).reshape((-1, 1))
x = x / x_l2

w_l2 = (np.sum(w ** 2, axis=0) ** (1 / 2)).reshape((1, -1))
w = w / w_l2


print('x= \n', x, '\n')
temp = x[0]
print(temp)
print('sum_x[0]:', np.sum(temp ** 2))
print('w= \n', w)
temp = w[:, 0]
print(temp)
print('sum_w[:,0]:', np.sum(temp ** 2))

cos = np.dot(x, w)
print(cos)

# 找到需要改变的cos值
groundtruth_score = []
for i in range(cos.shape[0]):
    groundtruth_score.append(float(cos[i][y[i]]))
groundtruth_score = np.array(groundtruth_score).reshape((-1, 1))
print(groundtruth_score)

M = np.greater(groundtruth_score, m) * m
print(M)

one_hot_y = np.zeros(cos.shape)
for i in range(len(one_hot_y)):
    one_hot_y[i][y[i]] = 1.
print('one hot y = \n', one_hot_y, '\n')
cos_min_m = (cos - one_hot_y * M)
print('cos-m = \n', cos_min_m)
print('only change 1 value, from 0.4082 to 0.0582.')

exp_feature = np.exp(cos_min_m * s)
print(exp_feature)

sum_feature = np.sum(exp_feature, axis=1).reshape((-1, 1))
print(sum_feature)

feature_norm = exp_feature / sum_feature
feature_groundtruth_norm = np.sum(feature_norm * one_hot_y, axis=1).reshape((-1, 1))
print('feature_groundtruth_norm = \n', feature_groundtruth_norm)

# method v2
a = exp_feature * one_hot_y
print(a)
cos_theta_yi_m = np.sum(a, axis=1).reshape((-1, 1))
print(cos_theta_yi_m)
feature_groundtruth_norm = cos_theta_yi_m / sum_feature
print('feature_groundtruth_norm = \n', feature_groundtruth_norm)

loss = -1 / 3 * np.log(np.sum(feature_groundtruth_norm, axis=1)).reshape((-1, 1))
print(loss)

import tensorflow as tf

tf.keras.layers.Dense