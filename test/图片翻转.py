'''''
函数：DataAugment（）
函数功能：扩大数据量
输入参数：dir_path----图片库路径
'''
import copy
import os

import cv2


def DataAugment(dir_path):
    if not os.path.exists(dir_path):
        print
        u'路径不存在'
    else:
        dirs = os.listdir(dir_path)
        for subdir in dirs:
            if(subdir != 'pink'):
                continue
            sub_dir = dir_path + '/' + subdir
            files = os.listdir(sub_dir)
            # fileNum = len(files)
            # if fileNum > 25:
            #     continue
            num = 0
            for fr in files:
                try:
                    suff = fr.split('.')[1]
                    filename = sub_dir + '/' + fr
                    if (filename.endswith(".gif")):
                        continue
                    print(filename)
                    img = cv2.imread(filename)
                    size = img.shape  # 获得图像的形状
                    iLR = copy.deepcopy(img)  # 获得一个和原始图像相同的图像，注意这里要使用深度复制
                    h = size[0]
                    w = size[1]
                    for i in range(h):  # 元素循环
                        for j in range(w):
                            iLR[i, w - 1 - j] = img[i, j]  # 注意这里的公式没，是不是恍然大悟了（修改这里）
                    new_name = "%s/%09d.%s" % (sub_dir, num, suff)
                    num += 1
                    cv2.imwrite(new_name, iLR)
                    # cv2.imshow('image',iLR)
                    # cv2.waitKey(0)
                except AttributeError:
                    print('AttributeError: ', filename)



DataAugment('D:/retrain/retrain_data_color/backpack/train_data')
