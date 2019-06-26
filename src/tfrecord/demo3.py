import tensorflow as tf
from multiprocessing import cpu_count


# image_classes = 85161
image_classes = 10533
img_size = [112, 96]

def img_preprocess(img):
    img_pre = (img - 127.5) * 0.0078125
    return img_pre

def _parse_function(example_proto):
    dics = {
        'label': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None),
        'image': tf.FixedLenFeature(shape=(), dtype=tf.string)
    }
    parsed_example = tf.parse_single_example(example_proto, dics)
    image = tf.decode_raw(parsed_example['image'], tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [img_size[0], img_size[1], 3])

    image_random_flip = tf.image.random_flip_left_right(image)
    image_pre_process = img_preprocess(image_random_flip)

    label = parsed_example['label']
    label_one_hot = tf.one_hot(label, image_classes, 1, 0)
    label = tf.reshape(label_one_hot, (image_classes,))
    return image_pre_process, label


def data_set(batch_size):
    # dataset = tf.data.TFRecordDataset(['D:\panlc\Data\\tfrecords\\czy_face_img_112_96_v2_[size-(112, 96)_init-11109_classes-242_num-24628].tfrecords'])
    dataset = tf.data.TFRecordDataset(['G:\\CASIA-WebFace_align_112_96_method2.tfrecords'])

    dataset = dataset.shuffle(buffer_size=30000)
    dataset = dataset.map(map_func=_parse_function, num_parallel_calls=cpu_count())
    dataset = dataset.batch(batch_size)

    # dataset = dataset.apply(tf.data.experimental.map_and_batch(
    #     map_func=_parse_function,
    #     batch_size=batch_size,
    #     num_parallel_calls=count_cpu))

    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=1000)
    return dataset


dataset = data_set(256 * 4)
iterator = dataset.make_one_shot_iterator()
data = iterator.get_next()

with tf.Session() as sess:
    for i in range(20000):
        d, ll = sess.run(data)
        print('========================i:', i, d.shape)
