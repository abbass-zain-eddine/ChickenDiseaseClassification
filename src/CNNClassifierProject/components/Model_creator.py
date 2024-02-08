import torch
import torch.nn as nn
import torchvision.models as models

# Example model (replace this with your own model)
class Model(nn.Module):
    def __init__(self, backbone, custom_layers_types, custom_layers_params, num_classes):
        super(Model, self).__init__()
        # Load the pre-trained backbone model
        self.backbone = backbone
        # Define custom layers based on the provided types and parameters
        self.custom_layers = nn.ModuleList()
        for layer_type, layer_params in zip(custom_layers_types, custom_layers_params):
            self.custom_layers.append(layer_type(**layer_params))

    def forward(self, x):
        x = self.backbone(x)
        for layer in self.custom_layers:
            x = layer(x)
        return x


