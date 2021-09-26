# 你只需要将数据整合到名为datasetname的文件夹下，分images、annotations两个文件夹存储图像、标注, 并保证对应的图像与标注文件名相同
# 然后调用这个函数，它会帮你生成合适的数据存储格式以及必要的文件
import os
import math

# def generate_datafolder(path, ratio):
#     # eg: ratio = [0.7, 0.1, 0.2] = train : valid : test
#
#     root_dir = path
#     img_dir = path + '/images'
#     ann_dir = path + '/annotations'
#
#     assert os.path.exists(root_dir), 'Dataset Folder not exist.'
#     assert os.path.exists(img_dir), 'You must build folder named images.'
#     assert os.path.exists(ann_dir), 'You must build folder named annotations.'
#
#     img_names = os.listdir(img_dir)
#     ann_names = os.listdir(ann_dir)
#
#     num = 0
#     file = open(path + path.split('/')[-1] + '.txt', 'w+')
#     for name in img_names:
#         pure_name = name.split('.')[0]
#         if pure_name + '.xml' in ann_names:
#             file.write(pure_name + '\n')
#             num += 1
#     file.close()
#     print('There are {} samples.'.format(num))
#
#     num_train = math.ceil(num * ratio[0])
#     num_valid = math.ceil(num * ratio[1])
#     num_test = num - num_train - num_valid
#     print('train: {}, valid: {}, test: {}'.format(num_train, num_valid, num_test))
#
#     os.mkdir(path + '/namesfiles')
#     file =


def from_divided_generate_namesfiles(root_dir):
    img_dir = root_dir + '/images'
    names_dir = root_dir + '/namesfiles'
    if not os.path.exists(names_dir):
        os.mkdir(names_dir)

    train_nameslist = [name.split('.')[0] for name in os.listdir(img_dir + '/train')]
    valid_nameslist = [name.split('.')[0] for name in os.listdir(img_dir + '/valid')]
    test_nameslist = [name.split('.')[0] for name in os.listdir(img_dir + '/test')]

    train_namesfile = open(root_dir + '/namesfiles/train.txt', 'w+')
    valid_namesfile = open(root_dir + '/namesfiles/valid.txt', 'w+')
    test_namesfile = open(root_dir + '/namesfiles/test.txt', 'w+')

    train_namesfile.writelines('\n'.join(train_nameslist))
    valid_namesfile.writelines('\n'.join(valid_nameslist))
    test_namesfile.writelines('\n'.join(test_nameslist))

    train_namesfile.close()
    valid_namesfile.close()
    test_namesfile.close()


if __name__ == '__main__':
    path = 'D:/Datasets'
    from_divided_generate_namesfiles(path)