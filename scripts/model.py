import torch
import torch.nn as nn
import torchvision.models as models

class ArtCRNN(nn.Module):
    def __init__(self, num_styles, num_artists):
        super(ArtCRNN, self).__init__()
        
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        
       
        self.rnn = nn.LSTM(input_size=512, hidden_size=256, batch_first=True, bidirectional=True)
        
         
        self.style_head = nn.Linear(512, num_styles)
        self.artist_head = nn.Linear(512, num_artists)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.flatten(2).permute(0, 2, 1) 
        
        rnn_out, _ = self.rnn(features)
        last_step = rnn_out[:, -1, :] 
        
        return self.style_head(last_step), self.artist_head(last_step)