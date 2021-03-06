import numpy as np
import tensorflow as tf


sequence = np.array([[1], [2], [ 4]])


def generator(steps):
    print(steps)
    for el in sequence:
        yield el


dataset = tf.data.Dataset().from_generator(lambda: generator(10), output_types=tf.float32, output_shapes=(tf.TensorShape([1,])))
dataset = dataset.repeat()
# ds = Dataset.from_generator(
#     gen, (tf.int64, tf.int64), (tf.TensorShape([]), tf.TensorShape([None])))


ite = dataset.make_one_shot_iterator()
data = ite.get_next()

with tf.Session() as sess:
    for i in range(4):
        x1 = sess.run(data)
        print(x1)
