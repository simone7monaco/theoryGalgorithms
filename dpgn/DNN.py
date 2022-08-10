import torch
from collections import OrderedDict

# Multi-layer Perceptron
class DNN(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        depth,
        drop_frac=0,
        act=torch.nn.Tanh,
        softmax=False
    ):
        super(DNN, self).__init__()
        
        layers = [('input', torch.nn.Linear(input_size, hidden_size))]
        layers.append(('input_activation', act()))
        layers.append(('input_dropout', 
                       torch.nn.Dropout(drop_frac)))
        for i in range(depth): 
            layers.append(
                ('hidden_%d' % i, torch.nn.Linear(hidden_size, hidden_size))
            )
            layers.append(('activation_%d' % i, act()))
            layers.append(('dropout_%d' % i, torch.nn.Dropout(drop_frac)))


        layers.append(('output', torch.nn.Linear(hidden_size, output_size)))
        if softmax == True:
            layers.append(('softmax', torch.nn.Softmax()))

        layerDict = OrderedDict(layers)
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out

    
class LakeLoss(torch.nn.Module):
    def __init__(self):
        super(LakeLoss, self).__init__()
    def forward(self, y_pred, y_true, udendiff):
        return torch.relu(udendiff)
    
    
class PhyLoss(torch.nn.Module):
    def __init__(self, phyloss, lam=.5):
        super(PhyLoss, self).__init__()
        self.mse = torch.nn.MSELoss()
        self.phyloss = phyloss
        self.lam = lam
        
    def forward(self, y_pred, y_true, udendiff):
        return self.mse(y_pred, y_true) + self.lam * torch.mean(self.phyloss(y_pred, y_true, udendiff))