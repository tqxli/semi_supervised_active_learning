from model.backbone import build_backbones
from base import BaseModel
from .detectors import FasterRCNN_custom
from .backbone import build_backbones

class FasterRCNN(BaseModel):
    def __init__(self, backbone_name='resnet50', fpn=True, pretrained=True):
        super(FasterRCNN, self).__init__()
        backbone = build_backbones(backbone_name=backbone_name, fpn=fpn, pretrained=pretrained)

        self.model = FasterRCNN_custom(backbone=backbone)

    def forward(self, x):
        return self.model(x)