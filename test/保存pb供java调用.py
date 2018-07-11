import tensorflow as tf

a = tf.constant(1.0, dtype=tf.float32, name='a')
b = tf.constant(2.3, dtype=tf.float32, name='b')
x = tf.placeholder(tf.float32, None, name='x')
c = tf.add_n([a, b, x], name='c')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    run = sess.run(c, feed_dict={x: 4.5})
    print(run)
    with tf.Graph().as_default() as graph:
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["c"])
        pbpath = './pb/expert-graph.pb'
        with tf.gfile.FastGFile(pbpath, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
