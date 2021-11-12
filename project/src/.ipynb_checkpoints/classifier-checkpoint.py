import torch
import torch.nn as nn
import torch.nn.functional as F
import wav2clip

def one_hot(labels, num_class):
    '''
    @params: labels in B x 1 where B is the batch size
    return one-hot vector in B x C where C is the number of classes
    '''
    B = labels.shape[0]
    one_hot = torch.zeros(B, num_class)
    one_hot.scatter_(1, labels.view(-1,1), 1)

    return one_hot


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

class SimpleClassifier(nn.Module):
    '''
    Mostly stolen from https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5
    '''
    def __init__(self, num_class=10, in_channel=1):
        super().__init__()
        self.num_class = num_class
        self.in_channel = in_channel

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channel, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(16),            
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        nn.init.kaiming_normal_(self.cnn[0].weight)
        nn.init.kaiming_normal_(self.cnn[3].weight)
        nn.init.kaiming_normal_(self.cnn[6].weight)
        nn.init.kaiming_normal_(self.cnn[9].weight)

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.predict_head = nn.Linear(in_features=64, out_features=num_class)


    def forward(self, x):
        B = x.shape[0]
        x = self.cnn(x)
        x = self.pool(x).view(B, 64)
        x = self.predict_head(x)
        return x


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
    
    
class Wav2CLIPClassifier(nn.Module):
    def __init__(self, input_ch=512, layer_num=2, classes_num=10, freeze_encoder=True):
        super(Wav2CLIPClassifier, self).__init__() 
        self.encoder = wav2clip.get_model()
        self.predictor = mlp(512, 2, 10)
        if freeze_encoder:
            self.encoder.require_grad = False
        
    def forward(self, x):
        ex = self.encoder(x)
        predicts = self.predictor(ex)
        return predicts
        
        
if __name__ == '__main__':
    model = Wav2CLIPClassifier()
    x = torch.rand(4, 64000)
    y = model(x)
    print(y.shape)
    
    