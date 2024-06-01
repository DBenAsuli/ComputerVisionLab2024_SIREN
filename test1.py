# Computer Vision Lab                  Project
# Dvir Ben Asuli                       318208816
# The Hebrew University of Jerusalem   2024

from main import *
from common import *

if __name__ == "__main__":
    folder_path = './Images/'
    test_reconstruct_folder("MLP", folder_path)
  #  test_reconstruct_folder("SIREN", folder_path)
    test_compare_folder('./Results/')

 #   mse_value, ssim_value = compare_images(image_path1, image_path2)

 #   print(f"MSE: {mse_value}")
 #   print(f"SSIM: {ssim_value}")