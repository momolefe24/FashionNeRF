import torch
import torch.nn as nn

class Net(nn.Module):
    '''
    Generates the NeRF neural network
    Args:
        num_layers: The number of MLP layers
        num_pos: The number of dimensions of positional encoding
    '''
    def __init__(self,num_layers,num_pos,encode_dims):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(99,64),nn.ReLU())
        self.dense_layers = nn.Sequential(nn.Linear(64,64),nn.ReLU())
        self.regulate_layers = nn.Sequential(nn.Linear(99+64,64),nn.ReLU())
        self.output_layer = nn.Linear(64,4)
        self.num_layers = num_layers
        self.layers = [64] * 8
        
    def forward(self,x):
        out = self.input_layer(x)
        for i in range(self.num_layers):
            out = self.dense_layers(out)
            if i % 4 == 0 and i > 0:
                out = torch.cat([out,x],dim=-1)
                out = self.regulate_layers(out)
        out = self.output_layer(out)
        return out
    