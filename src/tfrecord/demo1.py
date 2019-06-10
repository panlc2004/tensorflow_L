import tensorflow as tf
import numpy as np

tfrecords_filename = './tfrecords/train.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)  # 创建.tfrecord文件，准备写入

for i in range(100):
    img_raw = np.random.random_integers(0, 255, size=(7, 30))  # 创建7*30，取值在0-255之间随机数组
    img_raw = img_raw.tostring()
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
    writer.write(example.SerializeToString())

writer.close()
