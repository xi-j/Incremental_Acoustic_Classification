import torch
import torch.nn as nn
import torch.nn.functional as F


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

class mlp(nn.Module):
    def __init__(self, input_sz=512, layer_num=2, classes_num=10):
        
        super(mlp, self).__init__() 
        
        self.fc1 = nn.Linear(input_sz, input_sz, bias=True)
        self.fc2 = nn.Linear(input_sz, input_sz, bias=True)
        self.fc3 = nn.Linear(input_sz, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc2)
        init_layer(self.fc3)
 
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x