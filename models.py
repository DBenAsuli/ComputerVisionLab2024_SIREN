# Computer Vision Lab                  Project
# Dvir Ben Asuli                       318208816
# The Hebrew University of Jerusalem   2024

from common import *
import torch.nn as nn


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


# Neural network for MLP implicit representation
class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=526, output_dim=3, num_layers=10):
        super(MLP, self).__init__()
        layers = []

        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SIREN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=526, output_dim=3, num_layers=10, w0=30):
        super(SIREN, self).__init__()
        layers = []

        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(Sine(w0))
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

        # Initialization
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -1 / input_dim, 1 / input_dim)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)


class Sine(nn.Module):
    def __init__(self, w0=30):
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


# Neural network for SIREN  representation with ... FIXME
class SIREN2(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=526, output_dim=3, num_layers=10, w0=30):
        super(SIREN2, self).__init__()
        layers = []

        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(Sine(w0))
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x
