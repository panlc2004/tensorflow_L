import tensorflow as tf
import numpy as np

classes = 2
batch_size = 4
a = np.random.rand(batch_size, classes)
a1 = tf.reshape(a, [batch_size // 2, 2, classes])
y1, y2 = tf.unstack(a1, 2, 1)
n = tf.square(tf.subtract(y1, y2))
m = tf.add(y1, y2)
res = tf.divide(n, m)

x = [3.,4.]
y = tf.square(x)
#
# print(a)
# print('==================')
sess = tf.Session()
r, s = sess.run([y1, y2])
print(r)
print('++++++++++++++++++++')
print(s)

print('m:', sess.run(m))
print('n:', sess.run(n))
print('res:', sess.run(res))
print('y:', sess.run(y))
