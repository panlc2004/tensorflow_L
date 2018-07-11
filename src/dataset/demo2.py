import numpy as np
import tensorflow as tf

x = [1, 2, 3, 4, 5, 6]
y = [1.1, 2.1, 3.1, 4.1, 5.1, 6.1]


def _parsh(x, y):
    return x + 1, y + 1

def _apply(key_func, reduce_func, window_size):
    return x

dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.map(_parsh, 4).apply(_apply)
iterator = dataset.make_one_shot_iterator()
data = iterator.get_next()

with tf.Session() as sess:
    for i in range(6):
        print('i:', i)
        s = sess.run(data)
        print(s)
