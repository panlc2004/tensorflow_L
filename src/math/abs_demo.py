import tensorflow as tf

a = [1, -1, -2.1]
b = tf.abs(a)
sess = tf.Session()
print(sess.run(b))
