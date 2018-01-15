import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

batch_size = 100

n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x')
y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y')

w = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([1, 10]))
z = tf.matmul(x, w) + b
prediction = tf.nn.softmax(z)

loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction)

optimizer = tf.train.GradientDescentOptimizer(0.1)

train = optimizer.minimize(loss)

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, 'net/model_saver-20')
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
