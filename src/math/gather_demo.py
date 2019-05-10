import tensorflow as tf

alpha = 0.2
features = tf.constant([[4.0, 5.0], [13.0, 15.0], [6.0, 7.0], [1.0, 2.0], [18.0, 19.0]])


# 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
# 设置trainable=False是因为样本中心不是由梯度进行更新的
# centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
#                           initializer=tf.constant_initializer(0), trainable=False)
# 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
# 设置trainable=False是因为样本中心不是由梯度进行更新的
len_features = features.get_shape()[1]
centers = tf.get_variable('centers', [2, len_features], dtype=tf.float32,
                          initializer=tf.constant_initializer(0), trainable=False)

labels = tf.constant([[0], [1], [0], [0], [1]])
print(labels)
labels = tf.reshape(labels, [-1])
print(labels)

centers_batch = tf.gather(features, labels)
print(centers_batch)

loss = tf.nn.l2_loss(features - centers_batch)

# 当前mini-batch的特征值与它们对应的中心值之间的差
diff = centers_batch - features

# 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)

appear_times = tf.gather(unique_count, unique_idx)
appear_times = tf.reshape(appear_times, [-1, 1])

diff = diff / tf.cast((1 + appear_times), tf.float32)
diff = alpha * diff

centers_update_op = tf.scatter_sub(centers, labels, diff)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    c = sess.run(centers_batch)
    print(c)
    print(sess.run(loss))
    print(sess.run(diff))
    print('===============================')
    print('unique_label:', sess.run(unique_label))
    print('unique_idx:', sess.run(unique_idx))
    print('unique_count:', sess.run(unique_count))
    print('appear_times:', sess.run(appear_times))
    print('diff:', sess.run(diff))
    print('centers_update_op:', sess.run(centers_update_op))
