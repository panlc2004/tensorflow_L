import tensorflow as tf
import os
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt


# pb_path = 'D:/retrain/baggage_graph.pb'
# lable_file_path = 'D:/retrain/baggage_labels.txt'

pb_path = 'D:/tmp/output_graph_93.5.pb'
lable_file_path = 'D:/tmp/output_labels_93.5.txt'

lines = tf.gfile.GFile(lable_file_path).readlines()
uid_to_human = {}
# 一行一行读取数据
for uid, line in enumerate(lines):
    # 去掉换行符
    line = line.strip('\n')
    uid_to_human[uid] = line


def id_to_string(node_id):
    if node_id not in uid_to_human:
        return ''
    return uid_to_human[node_id]


# 创建一个图来存放google训练好的模型


with tf.gfile.FastGFile(pb_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    keep_prob = sess.graph.get_tensor_by_name('final_training_ops/keep_prob:0')
    # 遍历目录
    for root, dirs, files in os.walk('D:/retrain/retrain_data/backpack/'):
        for file in files:
            # 载入图片
            image_data = tf.gfile.FastGFile(os.path.join(root, file), 'rb').read()
            print('==================')
            print(image_data)
            print('==================')
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data, keep_prob:1.0})  # 图片格式是jpg格式
            print(predictions)

            predictions = np.squeeze(predictions)  # 把结果转为1维数据

            # 打印图片路径及名称
            image_path = os.path.join(root, file)
            print(image_path)

            print('================================')
            print(predictions)

            # 排序
            top_k = predictions.argsort()[::-1]
            # print(top_k)
            # for node_id in top_k:
            #     # 获取分类名称
            #     human_string = id_to_string(node_id)
            #     # 获取该分类的置信度
            #     score = predictions[node_id]
            #     print('%s (score = %.5f)' % (human_string, score))
            human_string = id_to_string(top_k[0])
            score = predictions[top_k[0]]
            print('%s (score = %.5f)' % (human_string, score))
            print()

            # 显示图片
            img = Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()
