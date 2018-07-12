import tensorflow as tf
sess = tf.InteractiveSession()
a = tf.constant([1.0, 3.0])
a = tf.Print(a, [a.shape], message="This is a: ")

print(sess.run(a))