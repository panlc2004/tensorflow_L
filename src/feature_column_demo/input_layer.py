# 参考文档：https://www.jianshu.com/p/fceb64c790f3
import tensorflow as tf
import numpy as np

price = {'price': np.zeros((4, 4)), 'test': np.ones((4, 4))}  # 4行样本

# column = tf.feature_column.numeric_column('price', normalizer_fn=lambda x: x + 2)
column = tf.feature_column.numeric_column(key='price', shape=(4,))
test = tf.feature_column.numeric_column(key='test', shape=(4,))
tensora = tf.feature_column.input_layer(price, [column])
print(tensora)
tensorb = tf.feature_column.input_layer(price, [test])
print(tensorb)
tensorab = tf.feature_column.input_layer(price, [column, test])
print(tensorab)
tensorab_rs = tf.reshape(tensorab, (-1, 2, 4))
print(tensorab_rs)

a = tf.split(tensorab_rs, 2, 1)

with tf.Session() as session:
    # print('tensora:', session.run([tensora]))
    # print('tensorb:', session.run([tensorb]))
    print('tensorab:', session.run([tensorab]))
    rs = session.run([tensorab_rs])
    print(tensorab_rs.shape)
    print('tensorab_rs:', rs)
    print('=====================')
    av = session.run([a])
    print(av[0])
