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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Main
if __name__ == "__main__":
    # For Memory Optimization and Warnings
    gc.collect()
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=UserWarning, message="A NumPy version.*")

    print("")
    print("* Please refer to the README.txt file in order to see instructions *")
    model_name = input("Enter the model name: ")
    image_name = input("Enter the image name inside \"Images\" directory: ")
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
        print("")
        print("Working on: \"" + image_name + "\" image")
        model = train(image, model, epochs=1000, lr=1e-3, img_name=image_name, model_name=model_name.upper())
        reconstructed_image = infer(model, image)

        plt.subplot(121)
        plt.imshow(image)
        plt.axis('off')
        plt.title('Original Image')

        plt.subplot(122)
        plt.imshow(reconstructed_image)
        plt.axis('off')
        plt.title("Reconstructed image with " + model_name.upper() + " Model")

        plt.show()
