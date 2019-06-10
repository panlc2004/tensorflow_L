import numpy as np


def cos(enm1, enm2):
    a = np.multiply(enm1, enm2)
    enm1_square = np.square(enm1)
    enm2_square = np.square(enm2)
    res = np.sum(a, axis=1) / np.sqrt(np.sum(enm1_square, axis=1) * np.sum(enm2_square, axis=1))
    # return res * 0.5 + 0.5
    return res


def cos2(enm1, enm2):
    sim = enm1.dot(enm2.T)
    norms = np.array([np.sqrt(np.diagonal(sim))])
    res = (sim / norms / norms.T)
    return res


def ac(y_true, y_pre):
    res = {'true_right': 0, 'true_wrong': 0, 'false_right': 0, 'false_wrong': 0}
    for i in range(len(y_true)):
        right = y_true[i] == y_pre[i]
        res_key = find_res_key(y_true[i], right)
        res[res_key] += 1
    return res


def find_res_key(y_true, right):
    if y_true:
        if right:
            return 'true_right'
        else:
            return 'true_wrong'
    else:
        if right:
            return 'false_right'
        else:
            return 'false_wrong'


m = np.asarray([[0, 1], [0, 1]])
m1 = np.square(m)
print(m1)
n = np.asarray([[0, 2], [1, 0]])
a = cos(m, n)
print(a)

x = [1, 2, 3, 4, 5]
y = np.less_equal(3, x)
print(y)
x1 = [1, 2, 3, 4, 1]
y1 = np.equal(x1, x)
print(y1)
y2 = ac(y, y1)
print(y2)

a = [1, 2, 3, 6, 2]
b = np.max(a)
c = np.argmax(a)
print(b)
print(c)
