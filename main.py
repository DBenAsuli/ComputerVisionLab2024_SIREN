# Computer Vision Lab                  Project
# Dvir Ben Asuli                       318208816
# The Hebrew University of Jerusalem   2024

import gc
import torch
import matplotlib.pyplot as plt

from common import *

# For Memory Optimization
gc.collect()

# Set device
device = torch.device('cpu')

# Main
if __name__ == "__main__":
    model_name = input("Please enter the preferred model name (MLP/SIREN): ")
    image_name = input("Please enter the (JPG) image name inside Images directory: ")
    image_path = './Images/' + image_name + '.JPG'
    image = load_image(image_path)
    H, W, C = image.shape

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
