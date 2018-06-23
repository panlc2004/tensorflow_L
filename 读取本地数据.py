import tensorflow as tf
from tensorflow.python.framework.errors_impl import OutOfRangeError

# 参考地址：https://blog.csdn.net/buptgshengod/article/details/72956846

folder = 'E:/SVN File/SourceCode/技术归化部/factnet_train/src/test/img/'

with tf.Session() as sess:
    # 我们要读三幅图片A.jpg, B.jpg, C.jpg
    filename = [folder + 'A.jpg', folder + 'B.jpg', folder + 'C.jpg']
    # string_input_producer会产生一个文件名队列
    filename_queue = tf.train.string_input_producer(filename, shuffle=True, num_epochs=5)
    # reader从文件名队列中读数据。对应的方法是reader.read
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    # tf.train.string_input_producer定义了一个epoch变量，要对它进行初始化
    tf.local_variables_initializer().run()
    # 使用start_queue_runners之后，才会开始填充队列
    threads = tf.train.start_queue_runners(sess=sess)
    k = tf.Print(key, [key])
    i = 0
    while True:
        i += 1
        try:
            image_data = sess.run(value)
        except OutOfRangeError as e:
            print(e.message)
            break
        with open('read/test %d.jpg' % i, 'wb') as f:
            f.write(image_data)
