import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
改进优化器，可以提高训练速度
'''

# 载入数据
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

batch_size = 100

n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 784], 'x')
y = tf.placeholder(tf.float32, [None, 10], 'y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
lr = tf.Variable(0.01, dtype=tf.float32)

# ==》权重初始化：正交化，可以提高训练效果
W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
b1 = tf.Variable(tf.zeros([500]) + 0.1)
z1 = tf.matmul(x, W1) + b1
a1 = tf.nn.relu(z1)
a1_drop = tf.nn.dropout(a1, keep_prob=keep_prob)

W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1))
b2 = tf.Variable(tf.zeros([300]) + 0.1)
z2 = tf.matmul(a1_drop, W2) + b2
a2 = tf.nn.relu(z2)
a2_drop = tf.nn.dropout(a2, keep_prob=keep_prob)

W3 = tf.Variable(tf.truncated_normal([300, 250], stddev=0.1))
b3 = tf.Variable(tf.zeros([250]) + 0.1)
z3 = tf.matmul(a2_drop, W3) + b3
a3 = tf.nn.relu(z3)
a3_drop = tf.nn.dropout(a3, keep_prob=keep_prob)

W4 = tf.Variable(tf.truncated_normal([250, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10]) + 0.1)
z4 = tf.matmul(a3_drop, W4) + b4
a4 = tf.nn.softmax(z4)
prediction = tf.nn.dropout(a4, keep_prob=keep_prob)

# 二阶损失函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# optimizer = tf.train.GradientDescentOptimizer(0.2)
# 修改优化器_此处修改为AdamOptimizer不再下降，无法训练，为什么？
# --> 学习速度设置过大，为0.001时，正常训练，0.01时，训练无效果
optimizer = tf.train.AdamOptimizer(lr)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        sess.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))
        for batch in range(n_batch):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            print('x_batch.shape:', x_batch.shape)
            print('y_batch.shape:', y_batch.shape)
            sess.run(train, feed_dict={x: x_batch, y: y_batch, keep_prob: 0.7})  # 设置keep_prob，缓解过拟合

        acc_test = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        acc_train = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
        learning_rate = sess.run(lr)
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc_test) + ", Train Accuracy " +
              str(acc_train) + ", Learning Rate= " + str(learning_rate))

