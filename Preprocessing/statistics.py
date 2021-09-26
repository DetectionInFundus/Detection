import os
import numpy as np
from xml.etree import ElementTree as ET
from matplotlib import pyplot as plt


def count_images_sizes(ann_dir):
    # 假设数据集存储格式已经整好了
    # 统计图像大小并做图，纵坐标轴为高度，横坐标轴为宽度

    train_ann_dir = ann_dir + '/train'
    valid_ann_dir = ann_dir + '/valid'
    test_ann_dir = ann_dir + '/test'

    ann_list = []
    ann_list.extend([(train_ann_dir + '/' + ann) for ann in os.listdir(train_ann_dir)])
    ann_list.extend([(valid_ann_dir + '/' + ann) for ann in os.listdir(valid_ann_dir)])
    ann_list.extend([(test_ann_dir + '/' + ann) for ann in os.listdir(test_ann_dir)])

    heights = []
    widths = []

    for ann in ann_list:
        tree = ET.parse(ann)
        img_size = tree.find('size')
        heights.append(float(img_size.find('height').text))
        widths.append(float(img_size.find('width').text))

    # print(len(heights))

    # colors = np.random.rand(len(heights)) + 0.1
    # plt.scatter(widths, heights, marker='o', alpha=0.3, c=colors)
    # plt.grid()
    # plt.show()

    # heights = [h / 600 for h in heights]
    # widths = [w / 600 for w in widths]
    #
    # colors = np.random.rand(len(heights)) + 0.1
    # plt.scatter(widths, heights, marker='o', alpha=0.3, c=colors)
    # plt.grid()
    # plt.show()

if __name__ == '__main__':
    count_images_sizes('D:/Datasets/annotations')