import torch
from torchvision.models import vgg16_bn, VGG16_BN_Weights

def get_vgg16_model(num_classes):
    vgg16_bn_weights = VGG16_BN_Weights.DEFAULT
    model = vgg16_bn(weights=vgg16_bn_weights)
    
    # Update the classifier
    num_features = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(num_features, num_classes)
    
    # Add Dropout to features
    new_features = []
    for layer in model.features:
        new_features.append(layer)
        if isinstance(layer, torch.nn.Conv2d):
            new_features.append(torch.nn.Dropout(p=0.3, inplace=True))
    
    model.features = torch.nn.Sequential(*new_features)
    return model
