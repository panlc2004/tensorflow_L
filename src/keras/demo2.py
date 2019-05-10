from keras import Sequential
from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input
from keras.layers import Dense, Activation, Dropout, Input, Reshape
from keras.optimizers import SGD

# 写神经网络
from keras_preprocessing.image import ImageDataGenerator

model = Sequential()
model.add(Input(shape=(28, 28,)))
model.add(Reshape(784))
model.add(Dense(500, input_shape=(784,)))  # 输入层，28*28=784
model.add(Activation('tanh'))  # 激活函数是tanh
model.add(Dropout(0.5))  # 采用50%的dropout

model.add(Dense(500))  # 隐藏层节点500个
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Dense(10))  # 输出结果是10个类别，所以维度是10
model.add(Activation('softmax'))  # 最后一层用softmax作为激活函数

model.summary()

# 加载数据
generator = ImageDataGenerator(rotation_range=2,
                               width_shift_range=0.02,
                               height_shift_range=0.02,
                               shear_range=0.01,
                               zoom_range=0.01,
                               preprocessing_function=preprocess_input)

train_data = generator.flow_from_directory("F:\Code\data\mnist\mnist_png\\training", target_size=(30, 30),
                                           batch_size=20)

# 开始训练
op = SGD(lr=0.001)

model.compile(loss='categorical_crossentropy', optimizer=op, metrics=['accuracy'])  # 使用交叉熵作为loss函数

model.fit(train_data, batch_size=20, epochs=50)
