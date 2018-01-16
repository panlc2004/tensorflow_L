import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, [None, 10], name="x")
y = tf.placeholder(tf.float32, [None, 5], name="y")

w_input = tf.placeholder(tf.float32, [10, 5], name="w")
w = tf.Variable(w_input)
b = tf.random_normal([1, 5], stddev=0.1)

z = tf.matmul(x, w) + b
prediction = tf.nn.relu(z)

loss = tf.reduce_mean(tf.square(y - prediction), name='loss')

with tf.Session() as sess:
    res = sess.run(prediction,
                   feed_dict={x: np.random.random([5, 10]), w: sess.run(tf.random_normal([10, 5], stddev=0.1))})
    print(res)
