import torch
import torchvision
from rpn.centernet_rpn import CenterNet

# backbone = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False).backbone

# rpn = CenterNet()

# roi_heads = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False).roi_heads




class CenterNet2(torch.nn.Module):
    def __init__(self, backbone, rpn, roi_heads, transforms):
        super(CenterNet2, self).__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.transforms = transforms

    def forward(self, images, targets):

        # 默认images经过预处理
        # 默认images是List[Tensor]类型，targets是[{'boxes':..., 'labels':..., 'image_id':..., 'area':...}]格式
        # transforms为Dict[str:function], 用于模块输入输出转化

        # resnet50_fpn_p6p7输入：标准的List[Tensor]格式
        # resnet50_fpn_p6p7输出：OderDict[层名:Tensor], 键值(层名)为['0', '1', '2', 'p6', 'p7']
        features = self.backbone(images)

        # 统一FPN输出的特征层的名字的格式
        features = self.transforms['unified_level_name_format'](features)
        # 将targets转化为detectron2中的模块需要的格式
        targets_for_detectron2 = self.transforms['standard_targets_2_detectron2_targets'](targets)
        # 将images转化为detectron2中的模块需要的格式
        images_for_detectron2 = self.transforms['list_tensor_2_imagelist'](images)

        # centernet_rpn输入：ImageList, OderDict[层名:Tensor]，List[Instances]
        # centernet_rpn输出：List[Instances], Dict[str:Tensor]。Instances的fields中包含以Boxes类型存储的预测框
        proposals, proposal_losses = self.rpn(images_for_detectron2, features, targets_for_detectron2)

        # 将detectron2格式的proposals转化为标准格式
        proposals = self.transforms['detectron2_proposal_2_standard_proposal'](proposals)

        # roi_heads输入：OderDict[层名:Tensor], List[Tensor], List[Tuple[h, w]], targets格式跟forward输入一样
        # roi_heads输出：Dict[str:Tensor], Dict[str:Tensor]
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        losses = {}
        losses.update(proposal_losses)
        losses.update(detector_losses)

        if self.training:
            return losses
        else:
            # 还差一步后处理,复原
            return detections