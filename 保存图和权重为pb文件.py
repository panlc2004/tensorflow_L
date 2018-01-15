import os
import shutil

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

export_dir = 'pb/test/'
# if os.path.exists(export_dir):
#     shutil.rmtree(export_dir)

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
batch_size = 100
n_batch = mnist.train.num_examples // batch_size

g = tf.Graph()
with g.as_default():
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y')

    w = tf.Variable(tf.random_normal([784, 10]), name='w')
    b = tf.Variable(tf.zeros([1, 10]), name='b')
    z = tf.matmul(x, w) + b
    prediction = tf.nn.softmax(z)

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction)

    optimizer = tf.train.GradientDescentOptimizer(0.1)

    train = optimizer.minimize(loss)

    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, 'net/model_saver-20')
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

        with tf.Graph().as_default() as graph:
            output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["accuracy"])
            pbpath = export_dir + 'expert-graph.pb'
            with tf.gfile.FastGFile(pbpath, 'wb') as f:
                f.write(output_graph_def.SerializeToString())

