import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

DIR = "tfrecord"


def read_TFRecords_test(name):
    filename = os.path.join(DIR, name + ".tfrecords")
    record_itr = tf.python_io.tf_record_iterator(path=filename)
    for r in record_itr:
        example = tf.train.Example()
        example.ParseFromString(r)

        label = example.features.feature["label"].int64_list.value[0]
        print("Label", label)
        image_bytes = example.features.feature["image"].bytes_list.value[0]
        img = np.fromstring(image_bytes, dtype=np.uint8).reshape(28, 28)
        print(img)
        plt.imshow(img, cmap="gray")
        plt.show()
        break

read_TFRecords_test("train")
