import tensorflow as tf

a = {'a': [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]}
b = [6, 7, 8, 9, 0]

ds = tf.data.Dataset.from_tensor_slices((a, b))
iterator = ds.make_one_shot_iterator()
data = iterator.get_next()

with tf.Session() as sess:
    for i in range(2):
        print('i:', i)
        s = sess.run(data)
        print(s)
