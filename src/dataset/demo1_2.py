import numpy as np
import tensorflow as tf

a = [[1, 2], [3, 4], [5, 6]]
b = [7, 8, 9]

sequence = np.array([['1', 1], ['2', 2], ['4', 4]])


def generator(steps):
    np.random.shuffle(sequence)
    for el in sequence:
        yield el[0], el[1]

def map_fun(data, label):
    print(data)
    print(label)
    return data, label

dataset = tf.data.Dataset.from_generator(lambda: generator(10), output_types=(tf.string, tf.int64))
dataset = dataset.map(map_fun).batch(2)
dataset = dataset.repeat()
# ds = Dataset.from_generator(
#     gen, (tf.int64, tf.int64), (tf.TensorShape([]), tf.TensorShape([None])))


ite = dataset.make_one_shot_iterator()
data = ite.get_next()

with tf.Session() as sess:
    for i in range(4):
        x1 = sess.run(data)
        print(x1)
