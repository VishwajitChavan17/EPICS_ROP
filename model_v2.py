import torch
import torch.nn as nn
from torchvision import models

def get_rop_model(num_classes=2, pretrained=True):
    """
    Returns an EfficientNet-B3 model tailored for ROP classification.
    """
    if pretrained:
        # Using the latest Weights API
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
        model = models.efficientnet_b3(weights=weights)
    else:
        model = models.efficientnet_b3()

    # Modify the classifier head
    # EfficientNet-B3 classifier is a Dropout followed by a Linear layer
    in_features = model.classifier[1].in_features
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )

    return model

if __name__ == "__main__":
    # Test if model builds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_rop_model().to(device)
    print(f"Model loaded on {device}")
    
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Should be [1, 2]
