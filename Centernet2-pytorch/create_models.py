def create_centernet2_baseline(num_classes):
    # backbone = resnet50_fpn_p6p7
    # rpn = centernet
    # roi_heads = from standard fasterrcnn

    # create backbone
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.ops.feature_pyramid_network import LastLevelP6P7
    backbone = resnet_fpn_backbone(
        'resnet50', pretrained=True, returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256)
    )

    # create rpn
    from rpn.centernet_rpn import CenterNet
    from util_from_cntn2.shape_spec import ShapeSpec
    feature_level_names = ['p3', 'p4', 'p5', 'p6', 'p7']
    shapes = {}
    for name in feature_level_names:
        shapes[name] = ShapeSpec(256, 0, 0, 0)
    rpn = CenterNet(shapes)

    # create roi_heads
    from torchvision.ops import MultiScaleRoIAlign
    from roi_origin.two_mpl_head import TwoMLPHead
    from roi_origin.fastrcnn_predictor import FastRCNNPredictor
    from roi_origin.roi_heads import RoIHeads
    box_roi_pool = MultiScaleRoIAlign(featmap_names=['p3', 'p4', 'p5', 'p6', 'p7'], output_size=[7, 7], sampling_ratio=2)
    # 一些参数
    out_channels = 256
    resolution = box_roi_pool.output_size[0]  # resultrion = 7
    representation_size = 1024
    # box_head, box_predictor
    box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)
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
    # roi_heads
    roi_heads = RoIHeads(box_roi_pool, box_head, box_predictor, box_fg_iou_thresh, box_bg_iou_thresh,
                         box_batch_size_per_image, box_positive_fraction, bbox_reg_weights, box_score_thresh,
                         box_nms_thresh, box_detections_per_img)

    # create model
    from model.centernet2 import CenterNet2
    from model.transform_utils import unified_level_name_format as ulnf
    from model.transform_utils import standard_targets_2_detectron2_targets as st2dt
    from model.transform_utils import detectron2_proposal_2_standard_proposal as dp2sp
    from model.transform_utils import list_tensor_2_imagelist as lt2imglst
    transforms = {
        'list_tensor_2_imagelist': lt2imglst,
        'unified_level_name_format': ulnf,
        'standard_targets_2_detectron2_targets': st2dt,
        'detectron2_proposal_2_standard_proposal': dp2sp
    }

    model = CenterNet2(backbone, rpn, roi_heads, transforms)
    return model


if __name__ == '__main__':
    MODEL = create_centernet2_baseline(4)
