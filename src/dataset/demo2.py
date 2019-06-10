import numpy as np
import tensorflow as tf

x = [1, 2, 3, 4, 5, 6]
y = [1.1, 2.1, 3.1, 4.1, 5.1, 6.1]


def _parsh(x, y):
    return x, y


def _apply(dataset):
    print('=========_apply=========')
    # x = [1.7, 2.7, 3.7, 4.7, 5.7, 6.7]
    # y = [1.1, 2.1, 3.1, 4.1, 5.1, 6.1]
    # dataset = tf.data.Dataset.from_tensor_slices((x, y))
    # dataset = dataset.map(_parsh, 4).repeat()
    return dataset


dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.map(_parsh, 4).apply(_apply).repeat()
iterator = dataset.make_one_shot_iterator()
data = iterator.get_next()

with tf.Session() as sess:
    for i in range(9):
        # print('i:', i)
        s = sess.run(data)
        print(s)
