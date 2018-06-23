import matplotlib.pyplot as plt;
import tensorflow as tf

# image_raw_data_jpg = tf.gfile.FastGFile('../img/img1.jpeg','rb').read()
image_raw_data_jpg = tf.read_file('../img/img1.png')

with tf.Session() as sess:
    img_data_jpg = tf.image.decode_image(image_raw_data_jpg) #图像解码
    img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.float32) #改变图像数据的类型

    plt.figure(1) #图像显示
    plt.imshow(img_data_jpg.eval())
    print(sess.run(img_data_jpg))
    plt.show()

