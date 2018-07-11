# 参考文档：https://www.jianshu.com/p/fceb64c790f3
# 哈希栏Hashed Column

import tensorflow as tf

colors = {'colors': ['green', 'red', 'blue', 'yellow', 'pink', 'blue', 'red', 'indigo']}

column = tf.feature_column.categorical_column_with_hash_bucket(
    key='colors',
    hash_bucket_size=5,
)

indicator = tf.feature_column.indicator_column(column)
tensor = tf.feature_column.input_layer(colors, [indicator])

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    print(session.run([tensor]))


# 我们注意到red和blue转化后都是一样的，yellow，indigo，pink也都一样，这很糟糕。
# 将hash_bucket_size箱子数量设置为10，这个问题可以得到解决。箱子数量的旋转很重要，越大获得的分类结果越精确。
# [array([[0., 0., 0., 0., 1.],#green
#         [1., 0., 0., 0., 0.],#red
#         [1., 0., 0., 0., 0.],#blue
#         [0., 1., 0., 0., 0.],#yellow
#         [0., 1., 0., 0., 0.],#pink
#         [1., 0., 0., 0., 0.],#blue
#         [1., 0., 0., 0., 0.],#red
#         [0., 1., 0., 0., 0.]], dtype=float32)]#indigo

