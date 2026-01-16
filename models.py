from torch import Tensor
import torch.nn as nn



class TransformerEncoder(nn.Sequential):
    def __init__(self,
                 config):
        num_heads=2
        super().__init__()
        d_model= config.num_electrodes
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):
        x=self.transformer_encoder(x)

        return x


class ClassificationHead(nn.Sequential):
    def __init__(self, config, n_classes):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(config.num_electrodes, n_classes)
        )

    def forward(self, x):
        
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)

        return out

class EEG_Transformer(nn.Sequential):
    def __init__(self, config):
        n_classes=config.n_classes

        super().__init__(
            TransformerEncoder(config),
            ClassificationHead(config, n_classes)
        )


class EEGNet(nn.Module):
    def __init__(self, config):
        super(EEGNet, self).__init__()
        self.T = 120
        
        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        
        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)
        
        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))
        
        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints. 
        self.fc1 = nn.Linear(4*2*10, 5)
        

    def forward(self, x):
        # Layer 1
        x=torch.unsqueeze(x,dim=1)

        # x=x.permute(0,1,3,2)
        x=x.transpose(3,2)

        batch_size = x.shape[0]

        x = F.elu(self.conv1(x))
       
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)

        # exit()
        x = self.padding1(x)
        
        x = F.elu(self.conv2(x))

        # exit()
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)
        # exit()
        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)
        # print('x:', x.shape)
        # exit()
        # FC Layer
        x=x.contiguous()
        x = x.view(batch_size, -1)
 
        x = F.sigmoid(self.fc1(x))

        return x




