# 参考文档：https://www.jianshu.com/p/fceb64c790f3
# 权重分类列WeightedCategoricalColumn
# 默认的CategoricalColumn所有分类的权重都是一样的，没有轻重主次。而权重分类特征列则可以为每个分类设置权重。


import tensorflow as tf
from tensorflow.python.feature_column.feature_column import _LazyBuilder

features = {'color': [['R'], ['A'], ['G'], ['B'], ['R']],
            'weight': [[1.0], [5.0], [4.0], [8.0], [3.0]]}

color_f_c = tf.feature_column.categorical_column_with_vocabulary_list(
    'color', ['R', 'G', 'B', 'A'], dtype=tf.string, default_value=-1
)

column = tf.feature_column.weighted_categorical_column(color_f_c, 'weight')

indicator = tf.feature_column.indicator_column(column)
tensor = tf.feature_column.input_layer(features, [indicator])

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    print(session.run([tensor]))


# 运行之后得到下面输出，权重改变了独热模式，不仅包含0或1，还带有权重值
#
# [array([[1., 0., 0., 0.],
#         [0., 0., 0., 5.],
#         [0., 4., 0., 0.],
#         [0., 0., 8., 0.],
#         [3., 0., 0., 0.]], dtype=float32)]
#