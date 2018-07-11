# import tensorflow as tf
# import numpy as np
#
# a = np.array([[1, 2, 3]])
# print(a)
# print(a.shape)
#
# b = tf.one_hot(a, depth=4)
# c = tf.argmax(b, 1)
#
# with tf.Session() as sess:
#     print(sess.run(b))
#     print(sess.run(c))
#



import numpy as np

data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(data)
print('data.shape:' , data.shape)
res = np.array([[1, 0, 1], [2, 1, 2]])
a = np.zeros([res.shape[0],data.shape[1], 3])
print(a.shape)
a[0,:] = res[0]
a[1,:] = res[1]
# print(a.shape)
# print(res[0].reshape(1,3))
# print(a)
print(data.shape)
c = np.subtract(data, a)
# print(c)

d = np.square(c)
print(d)
print('======================')
s = np.sum(d,axis=2)
print(s)
dist = np.sqrt(s)
print(dist)

print(dist.argsort()[::-1])

# [x, y, z]
# # [a,b]
# [[a, a, a],
#  [b, b, b]]
# [m, n]
