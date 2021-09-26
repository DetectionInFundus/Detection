import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from xml.etree import ElementTree as ET
from torchvision.transforms import functional as F


class MyDataSet(Dataset):

    def __init__(self, root_dir, dataset_type):
        self.root_dir = root_dir + '/'
        self.img_dir = root_dir + '/images/' + dataset_type + '/'
        self.ann_dir = root_dir + '/annotations/' + dataset_type + '/'
        self.names_dir = root_dir + '/namesfiles/' + dataset_type + '.txt'
        self.names_list = []
        self.class_dict = {'ma': 1, 'ex': 2, 'he': 3, 'se': 4}

        file = open(self.names_dir)
        for f in file:
            self.names_list.append(f)

    def __len__(self):
        return len(self.names_list)

    def __getitem__(self, idx):
        image_path = self.img_dir + self.names_list[idx].replace('\n', '') + '.jpg'
        assert os.path.isfile(image_path), image_path + 'does not exist!'

        image = Image.open(image_path)
        image = F.to_tensor(image)

        # (预处理接口)但目前为了减小计算量，打算用脚本先预处理图像，再加载到Dataset对象中来
        annotation_path = self.ann_dir + self.names_list[idx].replace('\n', '') + '.xml'
        assert os.path.exists(annotation_path), annotation_path + 'does not exist!'
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = []
        labels = []
        for obs in root.findall('object'):
            name = obs.find('name').text
            nums = obs.find('bndbox')
            x_min = float(nums.find('xmin').text)
            y_min = float(nums.find('ymin').text)
            x_max = float(nums.find('xmax').text)
            y_max = float(nums.find('ymax').text)

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(self.class_dict[name])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target['boxes'] = boxes
        target['lables'] = labels
        target['image_id'] = image_id
        target['area'] = area

        return image, target

    def coco_index(self, idx):
        """
            该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
            由于不用去读取图片，可大幅缩减统计时间

            Args:
                idx: 输入需要获取图像的索引
        """
        annotation_path = self.root_dir + self.names_list[idx] + '.xml'
        assert os.path.exists(annotation_path)
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        img_size = tree.find('size')
        height = img_size.find('height')
        width = img_size.find('width')

        boxes = []
        labels = []
        for obs in root.findall('object'):
            name = obs.find('name').text
            nums = obs.find('bndbox')
            x_min = float(nums.find('xmin').text)
            y_min = float(nums.find('ymin').text)
            x_max = float(nums.find('xmax').text)
            y_max = float(nums.find('ymax').text)

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(self.class_dict[name])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target['boxes'] = boxes
        target['lables'] = labels
        target['image_id'] = image_id
        target['area'] = area

        return (height, width), target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))