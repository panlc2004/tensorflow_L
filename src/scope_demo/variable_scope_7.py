import tensorflow as tf

x = tf.constant(1.0, tf.float32)
y = 2 * x
print(y)

with tf.variable_scope('test', 't', [x, y]):
    a = tf.get_variable('a', [1], dtype=tf.float32)
    b = tf.add_n([a + y])

print(tf.trainable_variables('test'))
