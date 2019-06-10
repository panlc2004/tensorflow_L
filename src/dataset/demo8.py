import numpy as np
import tensorflow as tf

x = [1.1, 2.1, 3.1, 4.1, 5.1, 6.1]
x = np.asarray(x)
x = np.reshape(x, (-1, 1))
y = [1, 2, 3, 4, 5, 6]
y = np.asarray(y)


# y = np.reshape(y, (-1, 1))

def _parsh(x, y):
    return x + 1, y + 1


def _apply(key_func):

    return x


def get_model():
    inputs = tf.keras.layers.Input(shape=[1, ])
    output = inputs
    model = tf.keras.Model(inputs, output)
    return model


dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.batch(3).repeat()
# dataset = dataset.map(_parsh, 4).apply(_apply)
iterator = dataset.make_one_shot_iterator()
data = iterator.get_next()

model = get_model()
model.summary()
opt = tf.keras.optimizers.SGD(lr=0.1, momentum=0.9)
model.compile(optimizer=opt, loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

sess = tf.Session()
tf.keras.backend.set_session(sess)


class MyCallBack(tf.keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):
        print('=================on_epoch_begin=====================')
        for i in range(2):
            s = sess.run(data)
            print(i, s)

    def on_epoch_end(self, epoch, logs=None):
        x = [1.2, 2.2, 3.2, 4.2, 5.2, 6.2]
        x = np.asarray(x)
        x = np.reshape(x, (-1, 1))
        y = [1, 2, 3, 4, 5, 6]
        y = np.asarray(y)
        global dataset, iterator, data
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.batch(3).repeat()
        iterator = dataset.make_one_shot_iterator()
        data = iterator.get_next()


back = MyCallBack()
back.set_model(model)
model.fit(dataset, epochs=2, steps_per_epoch=2, callbacks=[back])

#
# with tf.Session() as sess:
#     for i in range(2):
#         print('i:', i)
#         s = sess.run(data)
#         print(s)
