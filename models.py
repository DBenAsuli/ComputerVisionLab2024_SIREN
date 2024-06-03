# Computer Vision Lab                  Project
# Dvir Ben Asuli                       318208816
# The Hebrew University of Jerusalem   2024

import torch
from common import *
import torch.nn as nn


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


class Sine(nn.Module):
    def __init__(self, w0=30.0):
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

class SIREN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=3, num_layers=10, w0=30.0):
        super(SIREN, self).__init__()
        self.w0 = w0

        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(Sine(w0))

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            nn.init.uniform_(self.net[0].weight, -1 / self.net[0].in_features, 1 / self.net[0].in_features)
            nn.init.zeros_(self.net[0].bias)

            for layer in self.net:
                if isinstance(layer, nn.Linear):
                    nn.init.uniform_(layer.weight, -np.sqrt(6 / layer.in_features) / self.w0, np.sqrt(6 / layer.in_features) / self.w0)
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)


# Neural network for SIREN representation with alternating Sine and ReLU layers
class SIREN_HYBRID(SIREN):
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=3, num_layers=10, w0=30.0):
        super(SIREN, self).__init__()
        self.w0 = w0

        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(Sine(w0) if i % 2 == 0 else nn.ReLU(inplace=True))

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self.init_weights()


# Neural network for MLP implicit representation with one Sine Layer
class MLP_SINE(MLP):
    def __init__(self, input_dim=2, hidden_dim=526, output_dim=3, num_layers=10, w0=30):
        super(MLP_SINE, self).__init__()
        layers = []

        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(Sine(w0) if i == 0 else nn.ReLU(inplace=True))

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)
