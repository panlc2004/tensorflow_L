# 参考文档：https://www.jianshu.com/p/fceb64c790f3
import tensorflow as tf

price = {'price': [[1.], [2.], [3.], [4.]], 'test': [1, 2, 3]}  # 4行样本

column = tf.feature_column.numeric_column('price', normalizer_fn=lambda x: x + 2)
test = tf.feature_column.numeric_column('test')
print(column)
tensora = tf.feature_column.input_layer(price, [column])
tensorb = tf.feature_column.input_layer(price, [test])

with tf.Session() as session:
    print('tensora:', session.run([tensora]))
    print('tensorb:', session.run([tensorb]))
