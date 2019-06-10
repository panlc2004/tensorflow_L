import numpy as np
import tensorflow as tf

# max_value = tf.placeholder(tf.int64, shape=[])
# dataset = tf.data.Dataset.range(max_value)
max_value = tf.placeholder(tf.int64, shape=[None,])
dataset = tf.data.Dataset.from_tensor_slices((max_value))
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Initialize an iterator over a dataset with 10 elements.
with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={max_value: [0, 1, 2]})
    for i in range(3):
        value = sess.run(next_element)
        print(value)

    # Initialize the same iterator over a dataset with 100 elements.
    sess.run(iterator.initializer, feed_dict={max_value: [4, 5, 6]})
    for i in range(3):
        value = sess.run(next_element)
        print(value)


