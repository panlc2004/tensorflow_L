# 参考文档：https://www.jianshu.com/p/fceb64c790f3

# 分箱特征栏Bucketized column就是把一维的普通列表变成了二维矩阵，升维了！
import tensorflow as tf

years = {'years': [1999, 2013, 1987, 2005]}

years_fc = tf.feature_column.numeric_column('years')
# 我们把1980～now年份用3个边界(1990，2000，2010)划为4段
column = tf.feature_column.bucketized_column(years_fc, [1990, 2000, 2010])

tensor = tf.feature_column.input_layer(years, [column])

with tf.Session() as session:
    print(session.run([tensor]))

# print:
# [array([[ 0.,  1.,  0.,  0.], =>1999
#         [ 0.,  0.,  0.,  1.], =>2013
#         [ 1.,  0.,  0.,  0.],=>1987
#         [ 0.,  0.,  1.,  0.]]=>2005, dtype=float32)]
