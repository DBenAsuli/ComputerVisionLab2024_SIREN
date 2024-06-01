# Computer Vision Lab                  Project
# Dvir Ben Asuli                       318208816
# The Hebrew University of Jerusalem   2024

from main import *
import warnings
from common import *

# For Memory Optimization
gc.collect()
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, message="A NumPy version.*")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    folder_path = './Images/'
    num_of_epochs = 1000

    print("Reconstructing Images via MLP:")
 #   test_reconstruct_folder("MLP", folder_path, num_of_epochs=num_of_epochs)
    print("")

    print("Reconstructing Images via SIREN:")
    test_reconstruct_folder("SIREN", folder_path, num_of_epochs=num_of_epochs)
    print("")

    print("Comparing Reconstructions of MLP:")
    test_compare_folder('./Images/', './Results/MLP/')
    print("")

    print("Comparing Reconstructions of SIREN:")
    test_compare_folder('./Images/', './Results/SIREN/')
    print("")