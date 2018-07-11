# tf.variable_scope() 主要结合 tf.get_variable() 来使用，实现 变量共享。

# 这里是正确的打开方式~~~可以看出，name 参数才是对象的唯一标识
import tensorflow as tf

with tf.variable_scope('v_scope') as scope1:
    weights1 = tf.get_variable('weights', shape=[2, 3])
    bias1 = tf.get_variable('bias', shape=[3])
    with tf.variable_scope('v_scope_inner'):
        weights1_inner = tf.get_variable('weights', shape=[4, 5])

print(tf.get_variable('v_scope/weights'), shape=[2, 3])
print(weights1.name)
print(weights1.shape)
print(weights1_inner.name)
print(weights1_inner.shape)

# 下面来共享上面已经定义好的变量
# note: 在下面的 scope 中的变量必须已经定义过了，才能设置 reuse=True，否则会报错
with tf.variable_scope('v_scope', reuse=True) as scope2:
    weights2 = tf.get_variable('weights')
    with tf.variable_scope('v_scope_inner'):
        weights2_inner = tf.get_variable('weights')

print('weights2.name: ', weights2.name)
print('weights2.shape: ', weights2.shape)
print(weights2_inner.name)
print(weights2_inner.shape)
