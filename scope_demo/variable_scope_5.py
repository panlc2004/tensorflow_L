import tensorflow as tf

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
with tf.variable_scope('L1', reuse=False):
    L1 = tf.layers.conv2d(X, 64, [3, 3], reuse=False)
    L2 = tf.layers.max_pooling2d(L1, [2, 2], [2, 2])
    L3 = tf.layers.dropout(L2, 0.7, True)

print(L1)