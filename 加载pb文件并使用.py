from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x')
y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y')

g = tf.Graph()
output_graph_path = 'pb/test/expert-graph.pb'

with tf.Graph().as_default() as graph:
    with tf.gfile.FastGFile(output_graph_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(
            graph_def,
            name=''
        )

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        x = sess.graph.get_tensor_by_name("x:0")
        y = sess.graph.get_tensor_by_name("y:0")
        accuracy = sess.graph.get_tensor_by_name("accuracy:0")
        res = sess.run([init, accuracy], feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print(res)
