import torch
import torch.nn as nn
from utils import FourierFeatureTransform
import torch

class NeuralHighlighter(nn.Module):
    def __init__(self, depth, width, out_dim, input_dim=2, positional_encoding=False, sigma=5.0):
        super(NeuralHighlighter, self).__init__()
        layers = []
        if positional_encoding:
            layers.append(FourierFeatureTransform(input_dim, width, sigma))
            layers.append(nn.Linear(width * 2 + input_dim, width))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm([width]))
        else:
            layers.append(nn.Linear(input_dim, width))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm([width]))
        for i in range(depth):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm([width]))
        layers.append(nn.Linear(width, out_dim))
        layers.append(nn.Softmax(dim=1))

        self.mlp = nn.ModuleList(layers)
        print(self.mlp)
    
    def forward(self, x):
        for layer in self.mlp:
            x = layer(x)
        return x


class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.output_image = nn.Parameter(torch.rand(224*224,2))

    def forward(self,x):
        return self.output_image
