import torch
import torchvision
from model.centernet2 import CenterNet2
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

from rpn.centernet_rpn import CenterNet
from util_from_cntn2.shape_spec import ShapeSpec
from util_from_cntn2.instances import Instances
from util_from_cntn2.boxes import Boxes
from util_from_cntn2.image_list import ImageList

from torchvision.ops import MultiScaleRoIAlign
from roi_origin.two_mpl_head import TwoMLPHead
from roi_origin.fastrcnn_predictor import FastRCNNPredictor
from roi_origin.roi_heads import RoIHeads


def train():
    # BACKBONE=带FPN的ResNet50(从torchvision提取)。FPN有5层，前3层连ResNet50。
    backbone = resnet_fpn_backbone('resnet50', pretrained=True, returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256))

    # 随机生成一个Tensor作为图像输入
    img = torch.rand(1, 3, 600, 600)
    # detectron2要求的图像输入格式
    img_list = ImageList(img, [(600, 600)])

    # 随机生成几个标注框和标注
    box = torch.rand(11, 4)
    label = torch.randint(1, 4, (11, 1))
    # detectron2要求的标注输入格式
    instances = Instances((600, 600))
    instances.gt_boxes = Boxes(box)
    instances.gt_classes = label
    print(instances)

    # 将图像输入backbone进行特征提取，得到features
    features = backbone(img_list.tensor)
    # print(features)
    # 定义了backbone输出的通道数为256, 模型需要将图像分为4类
    out_channels = 256
    num_classes = 4
    # backbone输出的5层即FPN输出的5层。更改一下输出层的名字。
    print(features.keys())  # output: odict_keys(['0', '1', '2', 'p6', 'p7'])
    features.update({'p3': features.pop('0')})
    features.update({'p4': features.pop('1')})
    features.update({'p5': features.pop('2')})
    print(features.keys())  # output: odict_keys(['p6', 'p7', 'p3', 'p4', 'p5'])
    # 将shape转化为detectron2规定的输入格式ShapeSepc。
    shapes = {}
    for k, v in features.items():
        size = v.size()
        # 这里可以基本证明传入得ShapeSpec只用了第一个值，即通道数
        # shapes[k] = ShapeSpec(size[1], size[2], size[3], size[0])
        shapes[k] = ShapeSpec(size[1], 0, 0, 0)
    for string, shape in shapes.items():
        print(string, shape)

    # RPN=CenterNet(从detectron2提取的结构)
    rpn = CenterNet(shapes)
    # 将图像、提取的特征、标注输入RPN，得到proposal和loss
    proposal, loss = rpn(img_list, features, [instances])
    print(proposal)
    print(loss)

    # 把格式转换为torchvision模型能接受的输入类型
    # proposal
    proposal_list = []
    for i in range(len(proposal)):
        proposal_list.append(proposal[i].proposal_boxes.tensor)
    print(proposal_list[0].size())
    # groundtruth
    boxes_for_tv = instances.gt_boxes.tensor
    labels_for_tv = instances.gt_classes.reshape(11)
    area_for_tv = instances.gt_boxes.area()
    idx_for_tv = [0]
    target = {}
    target['boxes'] = boxes_for_tv
    target['labels'] = labels_for_tv
    target['image_id'] = idx_for_tv
    target['area'] = area_for_tv
    print(labels_for_tv)

    # roi部分, roi_head有三个子结构：box_roi_pool, box_head, box_predictor, 建立roi_head对象除了三个子结构还需要若干参数
    # BOX_ROI_POOL=MultiScaleRoIAlign, 这一步会将原图像根据proposal切割成若干部分，然后摞在一起。如果有38个proposal就有38个通道
    box_roi_pool = MultiScaleRoIAlign(featmap_names=['p3', 'p4', 'p5', 'p6', 'p7'], output_size=[7, 7], sampling_ratio=2)

    # 参数作用待填补
    resolution = box_roi_pool.output_size[0]  # resultrion = 7
    representation_size = 1024
    # BOX_HEAD=TwoMLPHead(从torchvision提取)。 作用待填补。
    box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)
    # BOX_PREDICTOR=FastRCNNPredictor(从torchvision提取)。作用待填补。
    box_predictor = FastRCNNPredictor(representation_size, num_classes)

    # roi_head需要的参数。
    box_fg_iou_thresh = 0.5  # 前景iou
    box_bg_iou_thresh = 0.5  # 背景iou
    box_batch_size_per_image = 512  # 作用待填补
    box_positive_fraction = 0.25  # 作用待填补
    bbox_reg_weights = None  # 作用待填补
    box_score_thresh = 0.05  # 作用待填补
    box_nms_thresh = 0.5  # NMS
    box_detections_per_img = 100  # 作用待填补
    # ROI_HEADS=RoIHeads(从detectron2提取)。作用待填补。
    roi_heads = RoIHeads(box_roi_pool, box_head, box_predictor,box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,bbox_reg_weights,box_score_thresh, box_nms_thresh, box_detections_per_img)

    # 将backbone提取的特征、proposal、图像大小、标注输入roi_head, 得到预测结果result和损失losses
    result, losses = roi_heads(features, proposal_list, img_list.image_sizes, [target])
    print(result, losses)


if __name__ == '__main__':
    train()
