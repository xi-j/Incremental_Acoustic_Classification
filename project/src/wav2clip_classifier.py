import torch
import torch.nn as nn
import torch.nn.functional as F

from .wav2clip.encoder import ResNetExtractor

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

class w2c_classifier(nn.Module):
    def __init__(self, ckpt, scenario='frozen', num_layers=3, embedding_sz=512, classes_num=10):
        
        super(w2c_classifier, self).__init__() 

        assert scenario in ("frozen", "finetune")

        self.wav2clip_encoder = ResNetExtractor(
            checkpoint=ckpt,
            scenario=scenario,
            transform=True
        )

        if scenario == 'frozen':
            for param in self.wav2clip_encoder.parameters():
                param.requires_grad = False
        
        self.fc1 = nn.Linear(embedding_sz, embedding_sz, bias=True)
        self.fc2 = nn.Linear(embedding_sz, embedding_sz, bias=True)
        self.fc3 = nn.Linear(embedding_sz, classes_num, bias=True)

        mlp_list = []
        
        for i in range(num_layers - 1):
            mlp_list.extend([nn.Linear(embedding_sz, embedding_sz, bias=True), nn.ReLU()])

        mlp_list.extend([nn.Linear(embedding_sz, classes_num, bias=True)])

        self.mlp = nn.Sequential(*mlp_list)

        self.init_weight()

    def init_weight(self):
        for layer in self.mlp:
            if len(list(layer.parameters())) != 0:
                init_layer(layer)
 
    def forward(self, x):
        x = self.wav2clip_encoder(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x