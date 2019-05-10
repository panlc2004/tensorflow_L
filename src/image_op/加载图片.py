from datetime import datetime

import matplotlib.pyplot as plt;
import tensorflow as tf

# image_raw_data_jpg = tf.gfile.FastGFile('../img/img1.png','rb').read()
image_raw_data_jpg = tf.read_file('../../img/001.jpg')
with tf.Session() as sess:
    img_data_jpg = tf.image.decode_image(image_raw_data_jpg)  # 图像解码
    img_data_jpg.set_shape([None,None,None])
    img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.float32)  # 改变图像数据的类型
    # img_data_jpg = tf.image.resize_images(img_data_jpg, [100, 100])
    image = sess.run(img_data_jpg)
    h,w,c=image.shape
    plt.figure(1)  # 图像显示
    if c == 1:
        image = image.reshape(h,w)
    plt.imshow(image)
    print(sess.run(img_data_jpg))
    plt.show()

print(datetime.now().date())

s = 'aaaaa%sbbbbb'
a = s % ('1')
print(a)