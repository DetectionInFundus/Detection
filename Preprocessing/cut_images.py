import numpy as np
import math
import cv2


def split_image(image, hoped_size=(600, 600), name='img', suffix='.jpg', target_dir=''):
    # 将一张RGB图片image切为若干hoped_size大小的小图片
    # image为numpy三维(H, W, C)矩阵，hoped_size=(height, width)

    '''
        1）判断图片高宽是否能整除hoped_size
        2）对于不能整除的图片进行padding zero, 使其能被整除
        3）切分图片，按从左至右，从上至下编号，标注的坐标也要相应改变
        注：
        1）此函数只负责切一张图片
        2）会存储切分完的图像
    '''

    size = image.shape[:2]
    nsize = [0, 0]
    pad = [0, 0]
    flag = ((size[0] % hoped_size[0] == 0), (size[1] % hoped_size[1] == 0))
    if not flag[0] or not flag[1]:
        if not flag[0]:
            print('height={} 不能被整除'.format(size[0]))
            nsize[0] = math.ceil(float(size[0])/hoped_size[0])*hoped_size[0]
            pad[0] = nsize[0]-size[0]
        if not flag[1]:
            print('width={} 不能被整除'.format(size[1]))
            nsize[1] = math.ceil(float(size[1]) / hoped_size[1]) * hoped_size[1]
            pad[1] = nsize[1]-size[1]
        image = cv2.copyMakeBorder(image, 0, pad[0], 0, pad[1], cv2.BORDER_CONSTANT, value=0)
    else:
        nsize[0] = size[0]
        nsize[1] = size[1]

    num = (nsize[0]//hoped_size[0], nsize[1]//hoped_size[1])
    print(num)

    name_fmt = '{}/{}-({},{}).{}'  # 原名-(h_num, w_num)
    for i in range(num[0]):
        for j in range(num[1]):
            h_1, h_2 = int(i*hoped_size[0]), int((i+1)*hoped_size[0])
            w_1, w_2 = int(j*hoped_size[1]), int((j+1)*hoped_size[1])
            img = image[h_1:h_2, w_1:w_2]
            cv2.imwrite(name_fmt.format(target_dir, name, i, j, suffix), img)


if __name__ == '__main__':
    img = cv2.imread('007-1853-100.tif')
    split_image(img, hoped_size=(600, 600), name='007-1853-100', suffix='.tif')

