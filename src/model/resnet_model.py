import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

def get_resnet18(num_classes: int = 2, dropout: float = 0.6, pretrained: bool = True):
    """
    ResNet-18 backbone with dropout + linear classification head.
    Dropout=0.6 helps against overfitting on small datasets.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )
    return model
