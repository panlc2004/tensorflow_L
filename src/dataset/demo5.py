import tensorflow as tf
import numpy as np
import math

dataset = tf.data.Dataset.range(10).batch(6).shuffle(10)
dataset = dataset.repeat(2)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    for i in range(4):
        value = sess.run(next_element)
        print(value)
