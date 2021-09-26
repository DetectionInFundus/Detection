import os
import shutil
import numpy as np
import cv2
from cut_images import split_image
from create_xml import getBbxAndNum, createXML_1, createXML_2, createXML_3


def split(root_dir, hoped_size=(600, 600)):
    '''
        1）切分图像
        2）切分segmentation groundtruth
        3）弃置没有病变的图像和其seg gt
        4）重新生成目标检测标注
        注：
        1）参数root_dir下分为images和annotations两个文件夹
        2）root_dir/annotations下又按四类病变分为MA/EX/HE/SE文件夹
        3）切分后的图片和标注仍存在root_dir下，图片用images_new文件夹，标注用annotations_new
    '''
    # 原数据集
    img_dir = root_dir + '/images'
    ann_dir = root_dir + '/annotations'

    # 新建目录，为切分做准备
    img_split_dir = root_dir + '/images_split'
    ann_split_dir = root_dir + '/annotations_split'

    os.mkdir(img_split_dir)
    os.mkdir(ann_split_dir)
    os.mkdir(ann_split_dir + '/MA')
    os.mkdir(ann_split_dir + '/EX')
    os.mkdir(ann_split_dir + '/HE')
    os.mkdir(ann_split_dir + '/SE')

    # 获取所有图片和标注的文件名，为切分做准备
    img_list = os.listdir(img_dir)
    ann_classes = os.listdir(ann_dir)
    ann_list = {}
    for ann_class in ann_classes:
        ann_list[ann_class] = os.listdir(ann_dir + '/' + ann_class)

    # 切分图片和标注，并存到相应的文件夹中
    print('split images')
    for img_path in img_list:
        img = cv2.imread(img_dir + '/' + img_path)
        split_image(
            img,
            hoped_size=hoped_size,
            name=img_path.split('.')[0],
            suffix=img_path.split('.')[1],
            target_dir=img_split_dir
        )
    print('split annotations')
    for ann_class, ann_pathes in ann_list.items():
        for ann_path in ann_pathes:
            ann = cv2.imread(ann_dir + '/' + ann_class + '/' + ann_path)
            split_image(
                ann,
                hoped_size=hoped_size,
                name=ann_path.split('.')[0],
                suffix=ann_path.split('.')[1],
                target_dir=ann_split_dir + '/' + ann_class
            )

    # 新建一些目录，为存储xml文件做准备
    img_new_dir = root_dir + '/images_new'
    ann_new_dir = root_dir + '/annotations_new'

    os.mkdir(img_new_dir)
    os.mkdir(ann_new_dir)

    # 获取切分后图像、标注的文件名
    img_split_list = os.listdir(img_split_dir)
    ann_split_list = {}
    for ann_class in ann_classes:
        ann_split_list[ann_class] = os.listdir(ann_split_dir + '/' + ann_class)

    # 生成有标注子图的XML文件
    for i in range(len(img_split_list)):
        # 获取该张图片的bbx和bbx数
        bbxs = []
        nums = []
        for ann_class in ann_classes:
            ann = cv2.imread(ann_split_dir + '/' + ann_class + '/' + img_split_list[i].split('.')[0] + '.tif')
            if ann is not None:
                bbx, num = getBbxAndNum(ann)
                bbxs.append(bbx)
                nums.append(num)
            else:
                bbxs.append(None)
                nums.append(0)

        # 弃用没有标注的子图
        if np.sum(nums) == 0:
            img_split_list[i] = img_split_list[i] + '.X'  # 给弃置的子图做一个标记，以便在之后的操作去除该子图
            continue

        # 生成文件的头部
        root = createXML_1(
            'DDR',
            img_split_list[i],
            size=(hoped_size[0], hoped_size[1], 3)
        )
        # 生成四类标注
        for j in range(len(ann_classes)):
            if bbxs[j] is not None and nums[j] != 0:
                root = createXML_2(root, bbxs[j], nums[j], ann_classes[j])
        # 生成XML文件
        createXML_3(root, img_split_list[i], ann_new_dir)

    # 将images_split文件夹下有用的图片移到images_new文件夹下，最后删除images_split和annotations_split文件夹
    for img_path in img_split_list:
        # 带.X的文件是弃用的，不移动
        if img_path.split('.')[-1] == 'X':
            continue
        # 只移动含有标注的子图
        shutil.move(img_split_dir + '/' + img_path, img_new_dir + '/' + img_path)
    # 删除中间文件及文件夹，images_split和annotations_split
    shutil.rmtree(img_split_dir)
    shutil.rmtree(ann_split_dir)


if __name__ == '__main__':
    root_dir = 'D:/Datasets/DDR-seg-test'
    split(root_dir)
