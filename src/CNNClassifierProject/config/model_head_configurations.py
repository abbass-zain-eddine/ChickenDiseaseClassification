import torch
import torch.nn as nn

classes=4

custom_layers_types = [nn.Flatten,
                        torch.nn.Linear,
                        nn.ReLU,
                        torch.nn.Dropout,
                        torch.nn.Linear,
                        torch.nn.ReLU,
                        torch.nn.Linear,
                        torch.nn.LogSoftmax]
custom_layers_params = [{},
                        {'in_features': 512,'out_features': 128},
                        {'inplace': True},
                        {'p':0.5},
                        {'in_features': 128,'out_features': 64},
                        {'inplace': True},
                        {'in_features': 64,'out_features': classes},
                        {'dim': 1}]