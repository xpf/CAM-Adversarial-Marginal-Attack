import torch.nn as nn
from torchvision.models import resnet50, resnet101
from torchvision.models import wide_resnet50_2, wide_resnet101_2
from torchvision.models import resnext50_32x4d, resnext101_32x8d

MODELS = {
    'resnet50': resnet50, 'resnet101': resnet101,
    'wide_resnet50_2': wide_resnet50_2, 'wide_resnet101_2': wide_resnet101_2,
    'resnext50_32x4d': resnext50_32x4d, 'resnext101_32x8d': resnext101_32x8d,
}


def build_separated_model(model_name='resnet50'):
    assert model_name in MODELS.keys()
    model = MODELS[model_name](pretrained=True)
    extractor = nn.Sequential(*list(model.children())[:-2])
    pooling = nn.Sequential(*list(model.children())[-2:-1])
    classifier = nn.Sequential(*list(model.children())[-1:])
    return extractor, pooling, classifier
