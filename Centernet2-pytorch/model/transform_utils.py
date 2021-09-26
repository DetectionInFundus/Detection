def unified_level_name_format(features):
    # odict_keys(['0', '1', '2', 'p6', 'p7']) -> odict_keys(['p6', 'p7', 'p3', 'p4', 'p5'])
    features.update({'p3': features.pop('0')})
    features.update({'p4': features.pop('1')})
    features.update({'p5': features.pop('2')})
    return features


def standard_targets_2_detectron2_targets(targets, image_size=(600, 600)):
    # 标准targets格式转化为detectron2模块可用的格式
    from util_from_cntn2.instances import Instances
    from util_from_cntn2.boxes import Boxes
    instances = []
    for i in range(len(targets)):
        instance = Instances(image_size)
        instance.gt_boxes = Boxes(targets[i]['boxes'])
        instance.gt_classes = targets[i]['labels']
        instances.append(instance)
    return instances


def detectron2_proposal_2_standard_proposal(proposals):
    # detectron2格式proposal转化为标准格式proposal
    proposal_list = []
    for i in range(len(proposals)):
        proposal_list.append(proposals[i].proposal_boxes.tensor)
    return proposal_list

def list_tensor_2_imagelist(images):
    # List[Tensor] 转化为 ImageList
    from util_from_cntn2.image_list import ImageList
    img_list = ImageList.from_tensors(images)
    return img_list