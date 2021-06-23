import os
import pathlib


from src.models.modules.head import replace_head
from src.models.modules.mnistnet import MnistNet
from src.models.modules.cifar_resnet import (resnet20,
                                             resnet32,
                                             resnet44,
                                             resnet56,
                                             resnet110,
                                             resnet1202)

from src.models.modules.cifar_vgg import vgg_bn_drop, vgg_bn_drop_100