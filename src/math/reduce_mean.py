import tensorflow as tf
import tensorflow.contrib.eager as tce

tce.enable_eager_execution()

a1 = tf.constant([[1.0, 1.0, 1.0], [1.02, 1.02, 1.02]], tf.float32, name='a1')
a2 = tf.constant([[1.1, 1.1, 1.1], [1.12, 1.12, 1.12]], tf.float32, name='a2')

b1 = tf.constant([[2.0, 2.0, 2.0], [2.02, 2.02, 2.02]], tf.float32, name='b1')
b2 = tf.constant([[2.1, 2.1, 2.1], [2.12, 2.12, 1.12]], tf.float32, name='b2')

c1 = tf.constant([[3.0, 3.0, 3.0], [3.02, 3.02, 3.02]], tf.float32, name='c1')
c2 = tf.constant([[3.1, 3.1, 3.1], [3.12, 3.12, 3.12]], tf.float32, name='c2')

d1 = tf.constant([[4.0, 4.0, 4.0], [4.02, 4.02, 4.02]], tf.float32, name='d1')
d2 = tf.constant([[4.1, 4.1, 4.1], [4.12, 4.12, 4.12]], tf.float32, name='d2')

all = []
all.append((a1, a2))
# all.append((b1, b2))
all.append((None, b2))

all2 = []
all2.append((c1, a2))
# all2.append((d1, b2))
all2.append((None, b2))

iin = []
iin.append(all)
iin.append(all2)

print(iin)

print('=============================')

# b = tf.reduce_mean(a, axis=0)
average_grads = []
for grad_and_vars in zip(*iin):
    grads = []
    for g, m in grad_and_vars:
        if g is not None:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

    if len(grad) > 0:
        grad = tf.concat(axis=0, values=grads)
        print('grad: ', grad)
        grad = tf.reduce_mean(grad, 0)
        print('grad_mean: ', grad)

        v = grad_and_vars[0][1]
    if len(grad) == 0:
        grad = None
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

print('average_grads: ', average_grads)

print('===========================================')

def average_gradients(grads):#grads:[[grad0, grad1,..], [grad0,grad1,..]..]
    averaged_grads = []
    for grads_per_var in zip(*grads):
        grads = tf.reduce_mean(grads_per_var, 0)
        averaged_grads.append((grads[0], grads[1]))
    return averaged_grads

x = average_gradients(iin)
print('x: ', x)
