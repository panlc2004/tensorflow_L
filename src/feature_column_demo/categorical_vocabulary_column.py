# 参考文档：https://www.jianshu.com/p/fceb64c790f3

# 分类词汇列

import tensorflow as tf

pets = {'pets': ['rabbit', 'pig', 'dog', 'mouse', 'cat']}

# categorical_column_with_vocabulary_list： 将一个单词列表生成为分类词汇特征列

# num_ovv_buckets，Out-Of-Vocabulary，如果数据里面的某个单词没有对应的箱子，
# 比如出现了老鼠mouse，那么就会在【箱子总数4～num_ovv_buckets+ 箱子总数=7】，
# 如果num_ovv=3,那么老鼠mouse会被标记为4～7中的某个数字，可能是5，也可能是4或6。num_ovv不可以是负数。

column = tf.feature_column.categorical_column_with_vocabulary_list(
    key='pets',
    vocabulary_list=['cat', 'dog', 'rabbit', 'pig'],
    dtype=tf.string,
    default_value=-1,
    num_oov_buckets=2)
indicator = tf.feature_column.indicator_column(column)

tensor = tf.feature_column.input_layer(pets, [indicator])

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    print(session.run([tensor]))


# 独热list 有7个元素，这是由于【猫狗兔子猪4个+num_oov_buckets】得到的。
# [array([[0., 0., 1., 0., 0., 0., 0.], #'rabbit'
#         [0., 0., 0., 1., 0., 0., 0.], #'pig'
#         [0., 1., 0., 0., 0., 0., 0.], #'dog'
#         [0., 0., 0., 0., 0., 1., 0.], #mouse
#         [1., 0., 0., 0., 0., 0., 0.]], dtype=float32)] #'cat'

