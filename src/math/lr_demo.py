import tensorflow as tf

global_step = tf.get_variable(
    'global_step', [],
    initializer=tf.constant_initializer(0), trainable=False)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10):
        lr = tf.train.exponential_decay(1.0, global_step, 5, 0.1, staircase=True)
        lr_d = sess.run(lr)
        print('lr: ', lr_d)
        print(sess.run(global_step))
