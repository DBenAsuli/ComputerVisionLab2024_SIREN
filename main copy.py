# Computer Vision Lab                  Project
# Dvir Ben Asuli                       318208816
# The Hebrew University of Jerusalem   2024

import os
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.optim as optim
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the image
def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = np.array(image) / 255.0
    return image

# Save the image
def save_image(image, path):
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(path)

# Create coordinate grid
def create_coordinates_grid(H, W):
    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    xv, yv = np.meshgrid(x, y)
    coordiantes = np.stack([xv, yv], axis=-1)
    coordiantes = coordiantes.reshape(-1, 2)
    return coordiantes


# Neural network for MLP implicit representation
class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=3, num_layers=4):
        super(MLP, self).__init__()
        layers = []
        for i in range(num_layers):
        #    layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
    #    layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Linear(input_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Neural network for SIREN implicit representation
class SIREN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=3, num_layers=4, w0=30):
        super(SIREN, self).__init__()
        self.net = nn.ModuleList()
    #    self.net.append(nn.Linear(input_dim, hidden_dim))
        self.net.append(nn.Linear(input_dim, input_dim))

        self.net.append(Sine())

        for _ in range(num_layers - 1):
            self.net.append(nn.Linear(input_dim, input_dim))
       #     self.net.append(nn.Linear(hidden_dim, hidden_dim))
            self.net.append(Sine())

     #   self.net.append(nn.Linear(hidden_dim, output_dim))
        self.net.append(nn.Linear(input_dim, output_dim))
        self.net.apply(siren_init)

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x
    
# Training function for both networks
def train(image, model, epochs=1000, lr=1e-3):
    H, W, C = image.shape
    coordiantes = create_coordinates_grid(H, W)
    coordiantes = torch.FloatTensor(coordiantes).to(device)
    image = torch.FloatTensor(image).view(-1, C).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(coordiantes)
        loss = criterion(output, image)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    return model

# Inferring function for both networks
def infer_MLP(model, H, W):
    coordiantes = create_coordinates_grid(H, W)
    coordiantes = torch.FloatTensor(coordiantes).to(device)
    with torch.no_grad():
        model.eval()
        output = model(coordiantes)
    output = output.cpu().numpy()
    output = output.reshape(H, W, 3)
    return output

# Sine activation function
class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)
    
# SIREN initialization
def siren_init(m):
    if isinstance(m, nn.Linear):
        num_input = m.weight.size(-1)
        # Scale factor is a hyperparameter, often set to 30
        with torch.no_grad():
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

# Main
if __name__ == "__main__":
    model_name = input("Please enter the preferred model name (MLP/SIREN): ")
    image_name = 'Img1'  # FIXME make modular
    #image_name = input("Please enter the image name inside Images directory: ")
    image_path = './Images/' + image_name + '.JPG'
    image = load_image(image_path)
    H, W, C = image.shape

    if model_name.lower() == "mlp":
        model = MLP().to(device)
        model_chosen = 1
    elif model_name.lower() == "siren":
        model = SIREN().to(device)
        model_chosen = 1
    else:
        print("Invalid model name")
        pass

    if model_chosen:
    #    model = train(image, model, epochs=1000, lr=1e-3)
        model = train(image, model, epochs=5, lr=1e-3)

        reconstructed_image = infer_MLP(model, H, W)
        res_path = './Results/' + image_name + '.png'
        print("AAAAAAAAA")
        print(res_path)
        reconstructed_image_path = os.path.join(res_path)
        save_image(reconstructed_image, reconstructed_image_path)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(image)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Reconstructed Image')
        plt.imshow(reconstructed_image)
        plt.axis('off')

        plt.show()