from torch.functional import norm
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, mobilenet_backbone
from torchvision.models import resnet
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.ops import misc as misc_nn_ops

__all_resnets__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101',
                   'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']
__all_mobilenets__ = ['wide_resnet50_2', 'wide_resnet101_2', 
                      'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small']
__all__ = __all_resnets__ + __all_mobilenets__

def build_backbones(backbone_name='resnet50', 
                    fpn=True, 
                    pretrained=False, 
                    norm_layer=misc_nn_ops.FrozenBatchNorm2d,
                    trainable_layers=3):
    assert backbone_name in __all__, "The specified backbone is currently unsupported. Available:\n".join(__all__)

    if backbone_name in __all_resnets__:
        if fpn:
            return resnet_fpn_backbone(backbone_name=backbone_name, pretrained=pretrained, norm_layer=norm_layer, trainable_layers=trainable_layers)
        return resnet.__dict__[backbone_name](pretrained=pretrained, norm_layer=norm_layer)

    if backbone_name in __all_mobilenets__:
        return mobilenet_backbone(backbone_name=backbone_name, pretrained=pretrained, fpn=fpn, norm_layer=norm_layer, trainable_layers=trainable_layers)