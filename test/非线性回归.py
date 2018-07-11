import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 定义神经网络
# 第一层
w1 = tf.Variable(tf.random_normal([1, 10]))
b1 = tf.Variable(tf.zeros([1, 10]))
z1 = tf.matmul(x, w1) + b1
a1 = tf.nn.tanh(z1)
# 第二层
w2 = tf.Variable(tf.random_normal([10, 1]))
b2 = tf.Variable(tf.zeros([1, 1]))
z2 = tf.matmul(a1, w2) + b2
prediction = tf.nn.tanh(z2)

loss = tf.reduce_mean(tf.square(prediction - y))

optimizer = tf.train.GradientDescentOptimizer(0.1)

# train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    # 训练
    for step in range(2000):
        sess.run(train, feed_dict={x: x_data, y: y_data})
        # if step % 20 == 0:
        #     print(step, ':', loss)
    # 预测
    prediction_value = sess.run(prediction, feed_dict={x: x_data})
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()
