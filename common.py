# Computer Vision Lab                  Project
# Dvir Ben Asuli                       318208816
# The Hebrew University of Jerusalem   2024

import gc
import os
import cv2
import torch
import warnings
import numpy as np
from models import *
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from torch.optim import Adam
import matplotlib.pyplot as plt
from skimage.transform import rescale
from skimage.metrics import structural_similarity as ssim

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load the image
def load_image(image_path):
    image = plt.imread(image_path) / 255.
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
    y = (y - im_mean[None]) / im_std[None]

    return x, y, im_mean, im_std


# Generate a grid of coordinates for SIREN reconstruction
def generate_grid_siren(resolution=256):
    lin = np.linspace(-1, 1, resolution)
    x, y = np.meshgrid(lin, lin)
    xy = np.stack([x, y], axis=-1)
    return xy.reshape(-1, 2), x.shape


# Training function for all networks
def train(image, model, epochs=1000, lr=1e-3, img_name="", model_name=""):
    x, y, im_mean, im_std = create_coordinate_grid(image)

    x = torch.FloatTensor(x).to(device)
    y = torch.FloatTensor(y).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    epochs_bar = tqdm(range(epochs))
    train_loss = []
    for epoch in epochs_bar:
        model.train()
        optimizer.zero_grad()
        output = model.net(x)
        loss = torch.mean((y - output) ** 2)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    plot_train_graph(train_loss=train_loss, img_name=img_name, model_name=model_name)
    return model


# Inferring function for all networks
def infer(model, image):
    x, y, im_mean, im_std = create_coordinate_grid(image)
    x = x.to(device)
    y = y.to(device)
    im_mean = im_mean.to(device)
    im_std = im_std.to(device)
    model.to(device)

    with torch.no_grad():
        output = torch.Tensor.cpu((model.net(x) * im_std[None] + im_mean[None]).reshape(image.shape))
        output = output.numpy().clip(0, 1)

    return output


def resize_image(image, target_size):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def compare_images(image_path1, image_path2):
    # Load the images
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    # Determine the size to which both images will be resized
    target_size = (min(image1.shape[1], image2.shape[1]), min(image1.shape[0], image2.shape[0]))

    # Resize both images to the target size
    image1_resized = resize_image(image1, target_size)
    image2_resized = resize_image(image2, target_size)

    # Convert the images to grayscale
    gray_image1 = cv2.cvtColor(image1_resized, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2_resized, cv2.COLOR_BGR2GRAY)

    # Compute MSE
    mse_value = np.mean((gray_image1 - gray_image2) ** 2)

    # Compute SSIM
    ssim_value, diff = ssim(gray_image1, gray_image2, full=True)

    return mse_value, ssim_value


def reconstruct_img(file_path, model_name, num_of_epochs):
    image = load_image(file_path)
    image_name = file_path.split(".")[1].split("/")[2]

    if model_name.upper() == "MLP":
        model = MLP().to(device)
        model_chosen = 1
    elif model_name.upper() == "SIREN":
        model = SIREN().to(device)
        model_chosen = 1
    elif model_name.upper() == "SIREN_HYBRID":
        model = SIREN_HYBRID().to(device)
        model_chosen = 1
    elif model_name.upper() == "SIREN_NARROW":
        model = SIREN(hidden_dim=128).to(device)
        model_chosen = 1
    elif model_name.upper() == "SIREN_WIDER":
        model = SIREN(hidden_dim=526).to(device)
        model_chosen = 1
    elif model_name.upper() == "SIREN_DEEPER":
        model = SIREN(num_layers=8).to(device)
        model_chosen = 1
    elif model_name.upper() == "SIREN_SHALLOW":
        model = SIREN(num_layers=3).to(device)
        model_chosen = 1
    elif model_name.upper() == "MLP_SINE":
        model = MLP_SINE().to(device)
        model_chosen = 1
    elif model_name.upper() == "MLP_SINE2":
        model = MLP_SINE2().to(device)
        model_chosen = 1
    elif model_name.upper() == "MLP_SINE3":
        model = MLP_SINE3().to(device)
        model_chosen = 1
    else:
        model_chosen = 0
        print("Invalid model name")
        pass

    if model_chosen:
        print("")
        print("Working on: \"" + image_name + "\" image")
        model = train(image, model, epochs=num_of_epochs, lr=1e-3, img_name=image_name, model_name=model_name.upper())
        reconstructed_image = infer(model, image)
        reconstructed_image_path = "".join(('./Results/', model_name, '/', image_name, '.png'))
        save_image(reconstructed_image, reconstructed_image_path)


def iterate_reconstruct_folder(folder_path, model_type, num_of_epochs):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        file_suffix = file_path.split(".")[-1]
        if os.path.isfile(file_path):
            if file_suffix.upper() == "JPG":
                reconstruct_img(file_path, model_type, num_of_epochs)


def iterate_compare_folder(folder_path1='./Images/', folder_path2="."):
    for filename1 in os.listdir(folder_path1):
        file_suffix = filename1.split(".")[-1]
        if file_suffix.upper() == "JPG":
            for filename2 in os.listdir(folder_path2):
                file_suffix = filename2.split(".")[-1]
                if file_suffix.upper() == "PNG":
                    if filename1.split(".")[0] == filename2.split(".")[0]:
                        mse_value, ssim_value = compare_images(folder_path1 + filename1,
                                                               folder_path2 + filename2)
                        output_name = filename1.split(".")[0].split("_")[0]

                        print("---------------")
                        print(f"Image: {output_name}")
                        print(f"MSE: {mse_value}")
                        print(f"SSIM: {ssim_value}")
                        print("")


def plot_train_graph(train_loss, img_name="", model_name=""):
    plt.plot(train_loss, label="Train Loss")
    plt.title(f'Train Loss of {img_name} Image with the {model_name} Model')
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    graph_image_path = "".join(('./Results/', model_name, '/Graphs/', f'Train_Loss_{img_name}_{model_name}.png'))
    plt.savefig(graph_image_path)
    plt.clf()


def test_reconstruct_folder(model_type, folder_path, num_of_epochs):
    iterate_reconstruct_folder(folder_path, model_type.upper(), num_of_epochs)


def test_compare_folder(folder_path1, folder_path2):
    iterate_compare_folder(folder_path1, folder_path2)
