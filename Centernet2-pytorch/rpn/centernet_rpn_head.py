import math
from typing import List
import torch
from torch import nn
from torch.nn import functional as F

from util_from_cntn2.shape_spec import ShapeSpec

__all__ = ["CenterNetHead"]


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class CenterNetHead(nn.Module):
    def __init__(self, input_shape: List[ShapeSpec]):
        super(CenterNetHead, self).__init__()
        # 一些由配置文件定义的类属性
        self.num_classes = 4
        self.with_agn_hm = True
        self.only_proposal = True
        self.out_kernel = 3
        # NORM = "GN"
        norm = 'GN'

        # 关于classification\boudingbox\share的配置，关于卷积层数和是否使用可变形卷积（不用，因为USE_DEFORMABLE = False）
        head_configs = {"cls": (0, False),
                        "bbox": (4, False),
                        "share": (0, False)}
        # 获取所有输入层的通道数
        in_channels = [s.channels for s in input_shape]
        # 验证所有输入层的通道数是否相同。必须相同，否则中断。
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        # 因为所有输入层通道数都相同，所以输入层通道数就可以确定了
        in_channels = in_channels[0]
        # 设定classification\boudingbox\share的输入通道数
        channels = {
            'cls': in_channels,
            'bbox': in_channels,
            'share': in_channels,
        }
        # 遍历head_configs
        for head in head_configs:
            # 用来存储卷积层
            tower = []
            # 卷积层数，是否使用可变性卷积
            num_convs, use_deformable = head_configs[head]
            # 通道数
            channel = channels[head]
            # 遍历所有卷积层
            for i in range(num_convs):
                # 如果是最后一层且使用可变卷积
                # if use_deformable and i == num_convs - 1:
                #     conv_func = DFConv2d
                # else:
                #     # 否则仅使用普通卷积
                #     conv_func = nn.Conv2d
                conv_func = nn.Conv2d

                # 向序列添加卷积层
                tower.append(conv_func(in_channels if i == 0 else channel, channel, kernel_size=3, stride=1,padding=1, bias=True))
                # 如果是groupnorm且通道数是32的倍数
                if norm == 'GN' and channel % 32 != 0:
                    tower.append(nn.GroupNorm(25, channel))
                # elif norm != '':
                #     tower.append(get_norm(norm, channel))

                # 有一次卷积，有一次ReLu
                tower.append(nn.ReLU())
            # 向model添加cls/bbx/share模块，之后分别用self.XXX调用
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))
        # bbx预测
        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=(self.out_kernel, self.out_kernel),
                                   stride=(1, 1), padding=self.out_kernel // 2
                                   )

        # 将input_shape大小的一串Scale类对象加入model,且可以训练
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in input_shape])

        # 初始化所有普通卷积层
        for modules in [
            self.cls_tower, self.bbox_tower,
            self.share_tower,
            self.bbox_pred,
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    # 使用从正态分布中提取的值填充输入张量（l.weight）
                    torch.nn.init.normal_(l.weight, std=0.01)
                    # 使l.bias=0
                    torch.nn.init.constant_(l.bias, 0)

        # 使bbox_pred.bias=8.0
        torch.nn.init.constant_(self.bbox_pred.bias, 8.)
        # PRIOR_PROB = 0.01
        prior_prob = 0.01
        # 用prior_prob计算出bias_value
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        # WITH_AGN_HM=True
        if self.with_agn_hm:
            self.agn_hm = nn.Conv2d(
                in_channels, 1, kernel_size=(self.out_kernel, self.out_kernel),
                stride=(1, 1), padding=self.out_kernel // 2
            )
            torch.nn.init.constant_(self.agn_hm.bias, bias_value)
            torch.nn.init.normal_(self.agn_hm.weight, std=0.01)

        # 肯定不走这一支，因为ONLY_PROPOSAL=True
        # if not self.only_proposal:
        #     cls_kernel_size = self.out_kernel
        #     self.cls_logits = nn.Conv2d(
        #         in_channels, self.num_classes,
        #         kernel_size=cls_kernel_size,
        #         stride=1,
        #         padding=cls_kernel_size // 2,
        #     )
        #
        #     torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        #     torch.nn.init.normal_(self.cls_logits.weight, std=0.01)

    def forward(self, x):
        clss = []
        bbox_reg = []
        agn_hms = []
        for l, feature in enumerate(x):
            # centernet_head中分类和回归前面共同的特征提取部分，这也是centernet2的精髓之一
            feature = self.share_tower(feature)
            # 分类
            cls_tower = self.cls_tower(feature)
            # 框回归
            bbox_tower = self.bbox_tower(feature)

            # 走else，因为ONLY_PROPOSAL=True
            # if not self.only_proposal:
            #     clss.append(self.cls_logits(cls_tower))
            # else:
            #     clss.append(None)
            clss.append(None)

            # 走if，因为WITH_AGN_HM=True
            # if self.with_agn_hm:
            #     agn_hms.append(self.agn_hm(bbox_tower))
            # else:
            #     agn_hms.append(None)
            agn_hms.append(self.agn_hm(bbox_tower))

            # reg为bbx预测结果，然后乘以可训练的权重
            reg = self.bbox_pred(bbox_tower)
            reg = self.scales[l](reg)
            # 使reg经过relu函数
            bbox_reg.append(F.relu(reg))

        return clss, bbox_reg, agn_hms