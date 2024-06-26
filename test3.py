# Computer Vision Lab                  Project
# Dvir Ben Asuli                       318208816
# The Hebrew University of Jerusalem   2024

import warnings
from common import *

# For Memory Optimization
gc.collect()
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=Warning, message="A NumPy version.*")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    folder_path = './Images/'
    num_of_epochs = 1000

    print("Reconstructing Images via SIREN_WIDER:")
    test_reconstruct_folder("SIREN_WIDER", folder_path, num_of_epochs=num_of_epochs)
    print("")

    print("Comparing Reconstructions of SIREN_WIDER:")
    test_compare_folder(folder_path, './Results/SIREN_WIDER/')
    print("")

    print("Reconstructing Images via SIREN_NARROW:")
    test_reconstruct_folder("SIREN_NARROW", folder_path, num_of_epochs=num_of_epochs)
    print("")

    print("Comparing Reconstructions of SIREN_NARROW:")
    test_compare_folder(folder_path, './Results/SIREN_NARROW/')
    print("")
