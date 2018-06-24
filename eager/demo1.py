import tensorflow as tf
import tensorflow.contrib.eager as tfe

tfe.enable_eager_execution()

x = [[2., 3.]]
# 数元素各自相乘,维度必须相等
m = tf.multiply(x, x)

# 矩阵相乘 第一个矩阵的列数（column）等于第二个矩阵的行数（row）
# n = tf.matmul(x, x)
print(m)
# print(n)


a = tf.constant(12)
counter = 0
while not tf.equal(a, 1):
    if tf.equal(a % 2, 0):
        a = a / 2
    else:
        a = 3 * a + 1
    print(a)

print("===========================")


def square(x):
    return tf.multiply(x, x)


# gradients_function用于计算输入的 square() 偏导数
grad = tfe.gradients_function(square)
print(square(3.))  # [9.]
print(grad(3.))  # [6.]
print(grad(2.))  # [4.]

# 同样的 gradient_function 调用可用于计算 square() 的二阶导数。
gradgrad = tfe.gradients_function(lambda x: grad(x)[0])
print(gradgrad(3.))  # [2.]
