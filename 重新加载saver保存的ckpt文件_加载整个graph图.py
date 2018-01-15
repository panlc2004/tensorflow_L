import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

batch_size = 100

n_batch = mnist.train.num_examples // batch_size

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('net/model_saver-20.meta')
    new_saver.restore(sess, "net/model_saver-20")

    graph = tf.get_default_graph()

    accuracy = graph.get_tensor_by_name("accuracy:0")
    x = graph.get_operation_by_name('x').outputs[0]
    y = graph.get_operation_by_name('y').outputs[0]
    # x = graph.get_tensor_by_name("x:0")
    # y = graph.get_tensor_by_name("y:0")
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
