from typing import Tuple

import torchvision
from torch import nn

import backbone.base

from modeling.backbones.resnet import ResNet
from modeling.baseline import Baseline
import torch

class Custom_ResNet101(backbone.base.Base):

    def __init__(self, pretrained: bool):
        super().__init__(pretrained)

    def features(self) -> Tuple[nn.Module, nn.Module, int, int]:
        # resnet101 = torchvision.models.resnet101(pretrained=self._pretrained)
        # resnet101 = ResNet(layers=[3, 4, 23, 3])
        resnet101 = Baseline(num_classes=702, last_stride=2, model_path='', neck='bnneck', neck_feat='after', model_name='resnet101', pretrain_choice='')
        resnet101.load_state_dict(torch.load('/data_hdd/Jaewoo/pretrained_model/r101_duke_state_dict.pth'))

        children = list(resnet101.children())
        children_split = list(children[0].children())
        
        features = children_split[:-1]
        num_features_out = 1024

        hidden = children[-3]
        num_hidden_out = 2048

        # for parameters in [feature.parameters() for i, feature in enumerate(features) if i <= 4]:
        for parameters in [feature.parameters() for i, feature in enumerate(features)]: # freezing
            for parameter in parameters:
                parameter.requires_grad = False

        features = nn.Sequential(*features)

        return features, hidden, num_features_out, num_hidden_out
