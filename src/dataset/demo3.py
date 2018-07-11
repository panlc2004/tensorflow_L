import numpy as np
import tensorflow as tf


def _parsh(x):
    return x[0] + 1, x[1] + 2


x = tf.placeholder(tf.float32, shape=[None, 2])
dataset = tf.data.Dataset.from_tensor_slices(x)
dataset = dataset.map(_parsh)
iterator = dataset.make_initializable_iterator()
data = iterator.get_next()

# ds = np.random.sample((100,2))
ds = [[1, 1.1], [2, 2.2], [3, 3.3]]
ds2 = [[4, 4.1], [5, 5.2], [6, 6.3]]


def _parsh(x, y):
    return x + 1, y + 1


with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={x: ds})
    for i in range(3):
        print('i:', i)
        s = sess.run(data, feed_dict={x: ds})
        print(s)

    sess.run(iterator.initializer, feed_dict={x: ds2})
    for i in range(3):
        print('i:', i)
        s = sess.run(data, feed_dict={x: ds})
        print(s)
