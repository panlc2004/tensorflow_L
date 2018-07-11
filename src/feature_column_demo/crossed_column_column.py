# 参考文档：https://www.jianshu.com/p/fceb64c790f3
# 交叉列Crossed column

import tensorflow as tf

featrues = {
    'longtitude': [19, 61, 30, 9, 45],
    'latitude': [45, 40, 72, 81, 24]
}

longtitude = tf.feature_column.numeric_column('longtitude')
latitude = tf.feature_column.numeric_column('latitude')

longtitude_b_c = tf.feature_column.bucketized_column(longtitude, [33, 66])
latitude_b_c = tf.feature_column.bucketized_column(latitude, [33, 66])

column = tf.feature_column.crossed_column([longtitude_b_c, latitude_b_c], 12)

# 指示列Indicator Columns:指示列并不直接操作数据，但它可以把各种分类特征列转化成为input_layer()方法接受的特征列。
indicator = tf.feature_column.indicator_column(column)
tensor = tf.feature_column.input_layer(featrues, [indicator])

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    print(session.run([tensor]))
