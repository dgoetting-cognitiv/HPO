import torch
import torch.nn as nn


class Architecture:
    def __init__(self, layers: [int]):
        self._layers = layers

    def layers(self):
        return self._layers

    def __str__(self):
        return str(self._layers)

    def __repr__(self):
        return self.__str__()


class Network(nn.Module):
    def __init__(self, activation, arch: Architecture, dropout=None, reg=None):
        super(Network, self).__init__()
        self.neurons_per_layer = None
        self.hidden_layers = None
        self.layers = []
        self.init_layers(arch.layers())

        self.activation = activation
        self.dropout = dropout
        self.reg = reg

    def forward(self, x):
        for i in range(self.hidden_layers):
            layer = self.layers[i]
            x = self.activation(layer(x))

        return torch.sigmoid(self.layers[-1](x))

    def init_layers(self, arch):
        self.hidden_layers = len(arch)
        self.neurons_per_layer = arch
        prev = 2
        for i in range(self.hidden_layers):
            self.layers.append(nn.Linear(prev, arch[i]))
            prev = arch[i]
        self.layers.append(nn.Linear(prev, 1))

    def parameters(self, recurse: bool = True):
        for layer in self.layers:
            yield from layer.parameters(recurse=True)
