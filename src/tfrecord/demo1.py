import tensorflow as tf
import numpy as np


def write():
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


def read():
    def map_(example_proto):
        dics = {
            'label': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None),
            'img_raw': tf.FixedLenFeature(shape=(), dtype=tf.string)
        }
        parsed_example = tf.parse_single_example(example_proto, dics)
        image = tf.decode_raw(parsed_example['img_raw'],tf.int64)
        image = tf.reshape(image, [7,30])
        # img = parsed_example['img_raw']
        # label = parsed_example['label']
        # parsed_example['label'] = tf.reshape(parsed_example['label'], parsed_example['label'].shape)
        # parsed_example['img_raw'] = tf.reshape(parsed_example['img_raw'], parsed_example['img_raw'].shape)
        return parsed_example

    # filenames = ["test.tfrecord", "test.tfrecord"]
    filenames = ["./tfrecords/train.tfrecords"]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(map_)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
        for i in range(101):
            s = sess.run(next_element)
            print(s)


if __name__ == '__main__':
    write()
    # read()
