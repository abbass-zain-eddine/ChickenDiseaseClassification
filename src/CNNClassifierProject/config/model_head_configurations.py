import torch
import torch.nn as nn

nb_classes=4
custom_layers_types = [nn.Flatten,
                        torch.nn.Linear,
                        nn.ReLU,
                        torch.nn.Dropout,
                        torch.nn.Linear,
                        torch.nn.ReLU,
                        torch.nn.Linear,
                        torch.nn.ReLU,
                        torch.nn.Linear,
                        torch.nn.Softmax]
custom_layers_params = [{},
                        {'in_features': 1000,'out_features': 512},
                        {'inplace': True},
                        {'p':0.5},
                        {'in_features': 512,'out_features': 256},
                        {'inplace': True},
                        {'in_features': 256,'out_features': 64},
                        {'inplace': True},
                        {'in_features': 64,'out_features': nb_classes},
                        {'dim': 1}]

optimizer= torch.optim.Adam
loss_fn=nn.CrossEntropyLoss()
metrix=['accuracy']
learning_rate=5e-5
check_points_dir="check_points"
