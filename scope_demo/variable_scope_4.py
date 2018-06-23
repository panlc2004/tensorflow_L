import tensorflow as tf
import tensorflow.contrib.slim as slim

cols = []
inputs = tf.placeholder(tf.float32, [None, 299, 299, 3])
with tf.variable_scope('conv1') as scope1:
    a = slim.conv2d(inputs, 64, [3, 3], scope=scope1)
    a = slim.conv2d(a, 64, [3, 3], scope=scope1)
print(a.name)
print(a.shape)
# print(a1.name)
print(cols)
#
with tf.variable_scope('conv1', reuse=True):
    b = tf.Variable([1], name='Relu')
    c = tf.get_variable('Conv/Relu:0')
# print(b.name)
# print(b.shape)
