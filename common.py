# Computer Vision Lab                  Project
# Dvir Ben Asuli                       318208816
# The Hebrew University of Jerusalem   2024

import os
import cv2
import numpy as np

import main
from main import *
from skimage.metrics import structural_similarity as ssim


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

def reconstruct_img(file_path, model_name):
    image = load_image(file_path)
    image_name = file_path.split(".")[1].split("/")[2]
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
        model = train(image, model, epochs=2, lr=1e-3)

        reconstructed_image = infer(model, image)
        reconstructed_image_path = "".join(('./Results/', model_name, '/', image_name, '_Reconstructed', '.png'))
        save_image(reconstructed_image, reconstructed_image_path)

def iterate_reconstruct_folder(folder_path, model_type):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        file_suffix = file_path.split(".")[-1]
        if os.path.isfile(file_path):
            if file_suffix.upper() == "JPG":
                reconstruct_img(file_path, model_type)

def iterate_compare_folder(folder_path):
    for filename1 in os.listdir(folder_path + '/MLP'):
        file_suffix = filename1.split(".")[-1]
        if file_suffix.upper() == "PNG":
            for filename2 in os.listdir(folder_path + '/SIREN'):
                file_suffix = filename2.split(".")[-1]
                if file_suffix.upper() == "PNG":
                    if filename1 == filename2:
                        mse_value, ssim_value = compare_images(folder_path + '/MLP/' + filename1, folder_path + '/MLP/' + filename2)
                        output_name = filename1.split(".")[0].split("_")[0]

                        print("---------------")
                        print(f"Image: {output_name}")
                        print(f"MSE: {mse_value}")
                        print(f"SSIM: {ssim_value}")
                        print("")

def test_reconstruct_folder(model_type, folder_path):
    iterate_reconstruct_folder(folder_path, model_type.upper())

def test_compare_folder(folder_path):
    iterate_compare_folder(folder_path)


