# Computer Vision Lab                  Project
# Dvir Ben Asuli                       318208816
# The Hebrew University of Jerusalem   2024

import gc
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from torch.optim import Adam
import matplotlib.pyplot as plt
from skimage.transform import rescale

# For Memory Optimization
gc.collect()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the image
def load_image(image_path):
  image = plt.imread(image_path)/255.
  scale = 120/max(image.shape[:-1])
  scale = 0.1
  image = rescale(image, scale, channel_axis=-1)

  return image

# Save the image
def save_image(image, path):
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(path)

# Create coordinate grid
def create_coordinate_grid(im):
    xx, yy = np.meshgrid(np.linspace(-1, 1, im.shape[0]), np.linspace(-1, 1, im.shape[1]))
    pts = torch.from_numpy(np.stack([xx, yy]).T).float()
    # reshape position grid and image so that they're easy to use
    x, y = pts.clone().reshape(-1, 2), torch.from_numpy(im).float().reshape(-1, 3)

    # standardize image values by subtracting mean and dividing by standard deviation
    im_mean, im_std = torch.mean(y, dim=0), torch.std(y, dim=0)
    y = (y - im_mean[None])/im_std[None]

    return x, y, im_mean, im_std


# Neural network for MLP implicit representation
class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=526, output_dim=3, num_layers=8):
        super(MLP, self).__init__()
        layers = []

        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Neural network for SIREN implicit representation
class SIREN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=526, output_dim=3, num_layers=10, w0=30):
        super(SIREN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(Sine())

        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(Sine())
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x
    
# Training function for both networks
def train(image, model, epochs=1000, lr=1e-3):
    x, y, im_mean, im_std = create_coordinate_grid(image)

    x = torch.FloatTensor(x).to(device)
    y = torch.FloatTensor(y).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    epochs_bar = tqdm(range(epochs))

    for epoch in epochs_bar:
        model.train()
        optimizer.zero_grad()
        output = model.net(x)
        loss = torch.mean((y-output)**2)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    return model

# Inferring function for both networks
def infer(model, image):
    x, y, im_mean, im_std = create_coordinate_grid(image)

    with torch.no_grad():
        # generate image and reverse the standardization recon=net(x)*std + mean
        recon = (model.net(x) * im_std[None] + im_mean[None]).reshape(image.shape).numpy().clip(0, 1)

    return recon

class Sine(nn.Module):
    def __init__(self, omega_0: float=30):
        super().__init__()
        # todo: save the needed components

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass  # todo: implement this

# Main
if __name__ == "__main__":
    model_name = input("Please enter the preferred model name (MLP/SIREN): ")
    image_name = 'Orr'  # FIXME make modular
    #image_name = input("Please enter the (JPG) image name inside Images directory: ")
    image_path = './Images/' + image_name + '.JPG'
    image = load_image(image_path)
    H, W, C = image.shape

    if model_name.upper() == "MLP":
        model = MLP().to(device)
        model_chosen = 1
    elif model_name.upper() == "SIREN":
        model = SIREN().to(device)
        model_chosen = 1
    else:
        print("Invalid model name")
        pass

    if model_chosen:
        model = train(image, model, epochs=1000, lr=1e-3)

        reconstructed_image = infer(model, image)
        reconstructed_image_path = "".join(('./Results/', image_name, '_Reconstructed_', model_name.upper(), '.png'))
        save_image(reconstructed_image, reconstructed_image_path)

        plt.subplot(121)
        plt.imshow(image)
        plt.axis('off')
        plt.title('Original Image')

        plt.subplot(122)
        plt.imshow(reconstructed_image)
        plt.axis('off')
        plt.title("Reconstructed image with " + model_name.upper() + " Model")

        plt.show()