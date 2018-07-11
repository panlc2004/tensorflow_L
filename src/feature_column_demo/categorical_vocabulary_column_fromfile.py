# 参考文档：https://www.jianshu.com/p/fceb64c790f3
# 分类词汇列
# 单词有些时候会比较多，这时候我们可以直接从文件中读取文字列表：
import os
import tensorflow as tf

pets = {'pets': ['rabbit', 'pig', 'dog', 'mouse', 'cat']}

dir_path = os.path.dirname(os.path.realpath(__file__))
fc_path = os.path.join(dir_path, 'pets_fc.txt')

# 从文件中读取文字列表：
column = tf.feature_column.categorical_column_with_vocabulary_file(
    key="pets",
    vocabulary_file=fc_path,
    num_oov_buckets=0)

indicator = tf.feature_column.indicator_column(column)
tensor = tf.feature_column.input_layer(pets, [indicator])

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    print(session.run([tensor]))

# num_oov_buckets使用了0，并没有增加元素数量，但是也导致了mouse变成了全部是0的列表
# [array([[0., 0., 1., 0.], #rabbit
#         [0., 0., 0., 1.], #pig
#         [0., 1., 0., 0.], #dog
#         [0., 0., 0., 0.],#mosue
#         [1., 0., 0., 0.]], dtype=float32)] #cat

