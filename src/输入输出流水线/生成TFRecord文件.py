import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

DIR = "tfrecord"


# int64
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# bytes
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 创建TFRecord文件
def convert_to_TFRecords(dataset, name):
    images, labels = dataset.images, dataset.labels
    n_examples = dataset.num_examples

    filename = os.path.join(DIR, name + ".tfrecords")
    print("Wirting", filename)
    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(n_examples):
            image_bytes = images[index].tostring()
            label = labels[index]
            example = tf.train.Example(features=tf.train.Features(
                feature={"image": _bytes_feature(image_bytes), "label": _int64_feature(label)}))
            writer.write(example.SerializeToString())


mnist_datasets = input_data.read_data_sets("../MNIST_data", dtype=tf.uint8, reshape=False)
convert_to_TFRecords(mnist_datasets.train, "train")
convert_to_TFRecords(mnist_datasets.validation, "validation")
convert_to_TFRecords(mnist_datasets.test, "test")
