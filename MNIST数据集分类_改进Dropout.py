import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
设置keep_prob，缓解过拟合
'''

# 载入数据
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

batch_size = 100

n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 784], 'x')
y = tf.placeholder(tf.float32, [None, 10], 'y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
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
optimizer = tf.train.GradientDescentOptimizer(0.2)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x: x_batch, y: y_batch, keep_prob: 0.7})  # 设置keep_prob，缓解过拟合

        acc_test = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        acc_train = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc_test) + ", Train Accuracy " + str(acc_train))


"""
keep_prob = 1的结果：
Iter 0,Testing Accuracy 0.9168, Train Accuracy 0.919509
Iter 1,Testing Accuracy 0.9466, Train Accuracy 0.949291
Iter 2,Testing Accuracy 0.9547, Train Accuracy 0.960727
Iter 3,Testing Accuracy 0.9585, Train Accuracy 0.966
Iter 4,Testing Accuracy 0.9655, Train Accuracy 0.973909
Iter 5,Testing Accuracy 0.966, Train Accuracy 0.977564
Iter 6,Testing Accuracy 0.9676, Train Accuracy 0.980564
Iter 7,Testing Accuracy 0.9709, Train Accuracy 0.983982
Iter 8,Testing Accuracy 0.9709, Train Accuracy 0.984855
Iter 9,Testing Accuracy 0.9701, Train Accuracy 0.984836
Iter 10,Testing Accuracy 0.9728, Train Accuracy 0.987291
Iter 11,Testing Accuracy 0.9737, Train Accuracy 0.988218
Iter 12,Testing Accuracy 0.9728, Train Accuracy 0.989327
Iter 13,Testing Accuracy 0.9753, Train Accuracy 0.9898
Iter 14,Testing Accuracy 0.9743, Train Accuracy 0.990236
Iter 15,Testing Accuracy 0.9758, Train Accuracy 0.990873
Iter 16,Testing Accuracy 0.9755, Train Accuracy 0.991091
Iter 17,Testing Accuracy 0.9761, Train Accuracy 0.991455
Iter 18,Testing Accuracy 0.9753, Train Accuracy 0.991655
Iter 19,Testing Accuracy 0.976, Train Accuracy 0.991745
Iter 20,Testing Accuracy 0.9764, Train Accuracy 0.992055


keep_prob = 0.7的结果:
Iter 0,Testing Accuracy 0.9131, Train Accuracy 0.908364
Iter 1,Testing Accuracy 0.9332, Train Accuracy 0.929327
Iter 2,Testing Accuracy 0.9394, Train Accuracy 0.938
Iter 3,Testing Accuracy 0.9484, Train Accuracy 0.947836
Iter 4,Testing Accuracy 0.9538, Train Accuracy 0.954055
Iter 5,Testing Accuracy 0.9569, Train Accuracy 0.958273
Iter 6,Testing Accuracy 0.9593, Train Accuracy 0.961018
Iter 7,Testing Accuracy 0.962, Train Accuracy 0.963927
Iter 8,Testing Accuracy 0.9623, Train Accuracy 0.966709
Iter 9,Testing Accuracy 0.9659, Train Accuracy 0.969709
Iter 10,Testing Accuracy 0.9657, Train Accuracy 0.969
Iter 11,Testing Accuracy 0.9663, Train Accuracy 0.972055
Iter 12,Testing Accuracy 0.9684, Train Accuracy 0.971582
Iter 13,Testing Accuracy 0.969, Train Accuracy 0.974327
Iter 14,Testing Accuracy 0.9705, Train Accuracy 0.975236
Iter 15,Testing Accuracy 0.9718, Train Accuracy 0.976382
Iter 16,Testing Accuracy 0.9724, Train Accuracy 0.976836
Iter 17,Testing Accuracy 0.9731, Train Accuracy 0.9792
Iter 18,Testing Accuracy 0.9733, Train Accuracy 0.979218
Iter 19,Testing Accuracy 0.974, Train Accuracy 0.980727
Iter 20,Testing Accuracy 0.9745, Train Accuracy 0.981164


"""