import tensorflow as tf
import numpy as np


def decode(serialized_example):
    features = tf.parse_single_example(serialized_example,
                                       features={"image": tf.FixedLenFeature([], tf.string)
                                           , "label": tf.FixedLenFeature([], tf.int64)})
    image = tf.decode_raw(features["image"], tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [784])
    label = tf.cast(features["label"], tf.int64)
    return image, label


def normalize(image, label):
    image = image / 255.0 - 0.5
    return image, label


def create_dataset(filename, batch_size=64, is_shuffle=False, n_repeats=0):
    """create dataset for train and validation dataset"""
    dataset = tf.data.TFRecordDataset(filename)
    if n_repeats > 0:
        dataset = dataset.repeat(n_repeats)  # for train
    dataset = dataset.map(decode).map(normalize)  # decode and normalize
    if is_shuffle:
        dataset.shuffle(1000 + 3 * batch_size)

    dataset = dataset.batch(batch_size)
    return dataset


# 使用一个简单的全连接层网络来实现mnist的分类模型：
def model(inputs, hidden_size=(500, 500)):
    h1, h2 = hidden_size
    net = tf.layers.dense(inputs, h1, activation=tf.nn.relu)
    net = tf.layers.dense(net, h2, activation=tf.nn.relu)
    net = tf.layers.dense(net, 10, activation=None)
    return net


# 训练
n_train_examples = 55000
n_val_examples = 5000
n_epochs = 50
batch_size = 64
train_dataset = create_dataset("tfrecord/train.tfrecords", batch_size=batch_size, is_shuffle=True, n_repeats=n_epochs)
val_dataset = create_dataset("tfrecord/validation.tfrecords", batch_size=batch_size)

# 创建一个feedable iterator
handle = tf.placeholder(tf.string, [])
feed_iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
images, labels = feed_iterator.get_next()

train_iterator = train_dataset.make_one_shot_iterator()
val_iterator = val_dataset.make_initializable_iterator()

# 创建模型
logits = model(images, [500, 500])
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
loss = tf.reduce_mean(loss)

train_op = tf.train.AdamOptimizer(learning_rate=1e-04).minimize(loss)

predictions = tf.argmax(logits, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
with tf.Session() as sess:
    sess.run(init_op)  # 生成对应的handle
    train_handle = sess.run(train_iterator.string_handle())
    val_handle = sess.run(val_iterator.string_handle())

    # 训练
    for n in range(n_epochs):
        ls = []
        for i in range(n_train_examples // batch_size):
            _, l = sess.run([train_op, loss], feed_dict={handle: train_handle})
            ls.append(l)
        print("Epoch %d, train loss: %f" % (n, np.mean(ls)))
        if (n + 1) % 10 == 0:
            sess.run(val_iterator.initializer)
            accs = []
            for i in range(n_val_examples // batch_size):
                acc = sess.run(accuracy, feed_dict={handle: val_handle})
                accs.append(acc)
            print("\t validation accuracy: %f" % (np.mean(accs)))