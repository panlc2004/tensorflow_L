import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
batch_size = 50
n_batch = mnist.train.num_examples // batch_size

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')
    with tf.name_scope('x_image'):
        x_image = tf.reshape(x, [-1, 28, 28, 1], name='x_image')

with tf.name_scope('conv1'):
    with tf.name_scope('W'):
        initial = tf.truncated_normal([5, 5, 1, 32], stddev=0.1)  # 生成一个截断的正态分布,采样窗口为5*5，32个卷积核，从1个平面抽取特征
        W_con1 = tf.Variable(initial, name='W_con1')
    with tf.name_scope('b'):
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]), name='b_conv1')

    with tf.name_scope('conv2d'):
        Z1 = tf.nn.conv2d(x_image, W_con1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
    with tf.name_scope('relu'):
        A1 = tf.nn.relu(Z1, name='relu')
    with tf.name_scope('pool'):
        P1 = tf.nn.max_pool(A1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope('conv2'):
    with tf.name_scope('W'):
        initial = tf.truncated_normal([5, 5, 32, 64], stddev=0.1)  # 5*5的采样窗口，64个卷积核从32个平面抽取特征
        W_con2 = tf.Variable(initial, name='W_con2')
    with tf.name_scope('b'):
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]), name='b_conv2')

    with tf.name_scope('conv2d'):
        Z2 = tf.nn.conv2d(P1, W_con2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2
    with tf.name_scope('relu'):
        A2 = tf.nn.relu(Z2)
    with tf.name_scope('pool'):
        P2 = tf.nn.max_pool(A2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 28*28的图片第一次卷积后还是28*28，第一次池化后变为14*14(池化层可能不补0)
# 第二次卷积后为14*14，第二次池化后变为了7*7
# 进过上面操作后得到64张7*7的平面 ==》 shpae:[7*7*64]
# 计算公式：out_size = (n - f + 2*p)/s + 1

# 把池化层2的输出扁平化为1维 ==》 每一列是一个平面
with tf.name_scope('flat'):
    flag = tf.reshape(P2, [-1, 7 * 7 * 64], name='flat')

# 初始化第一个全连接层的权值
with tf.name_scope('fc1'):
    with tf.name_scope('W_fc1'):
        initial = tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1)
        W_fc1 = tf.Variable(initial, name='W_fc1')
    with tf.name_scope('b_fc1'):
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[1, 1024]), name='b_fc1')

    # 求第一个全连接层的输出
    with tf.name_scope('wx_plus_b'):
        z_fc1 = tf.matmul(flag, W_fc1) + b_fc1
    with tf.name_scope('relu'):
        a_fc1 = tf.nn.relu(z_fc1)

    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    with tf.name_scope('drop_fc1'):
        drop_fc1 = tf.nn.dropout(a_fc1, keep_prob, name='drop_fc1')

# 初始化第二个全连接层
with tf.name_scope('fc2'):
    with tf.name_scope('W_fc2'):
        initial = tf.truncated_normal([1024, 10], stddev=0.1)
        W_fc2 = tf.Variable(initial, name='W_fc2')
    with tf.name_scope('b_fc2'):
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[1, 10]), name='b_fc2')

    with tf.name_scope("wx_plus_b"):
        z_fc2 = tf.matmul(drop_fc1, W_fc2) + b_fc2

    with tf.name_scope("softmax"):
        prediction = tf.nn.softmax(z_fc2, name="prediction")

# 交叉熵代价函数
with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(
        input_tensor=tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction),
        name="cross_entropy"
    )
    tf.summary.scalar('cross_entropy', cross_entropy)

# 优化器
with tf.name_scope("train"):
    train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 求准备率
with tf.name_scope("accuracy"):
    correction_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

# 合并所有的summary
merged = tf.summary.merge_all()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter("logs/train", sess.graph)
    test_writer = tf.summary.FileWriter("logs/test", sess.graph)

    for epoch in range(20):
        for batch in range(batch_size):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x: x_batch, y: y_batch, keep_prob: 0.7})
            # 记录训练集计算的参数
            summary = sess.run(merged, feed_dict={x: x_batch, y: y_batch, keep_prob: 1.0})
            train_writer.add_summary(summary, epoch)
            # 记录测试集计算的参数
            summary = sess.run(merged, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            test_writer.add_summary(summary, epoch)

            test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            train_acc = sess.run(accuracy, feed_dict={x: x_batch, y: y_batch, keep_prob: 1.0})
            print("Iter " + str(epoch) + ", Testing Accuracy= " + str(test_acc) +
                  ", Training Accuracy= " + str(train_acc))

    saver.save(sess, "net/mnist_conv")
