import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

batch_size = 100

n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x')
y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y')

w = tf.Variable(tf.random_normal([784, 10]), name='w')
b = tf.Variable(tf.zeros([1, 10]), name='b')
z = tf.matmul(x, w) + b
prediction = tf.nn.softmax(z, name='prediction')

loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction)

optimizer = tf.train.GradientDescentOptimizer(0.1)

train = optimizer.minimize(loss, name='train')

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

init = tf.global_variables_initializer()

saver = tf.train.Saver()

tf.add_to_collection('accuracy_2', accuracy)

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(20):
        for batch in range(n_batch):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x: x_batch, y: y_batch})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("epoch: ", epoch, " accuracy: ", acc)

    saver.save(sess, 'net/model_saver', global_step=20)


