# Computer Vision Lab                  Project
# Dvir Ben Asuli                       318208816
# The Hebrew University of Jerusalem   2024

from main import *
from common import *

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    folder_path = './Images/'
    num_of_epochs = 1000

    # For Memory Optimization and Warnings
    gc.collect()
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=UserWarning, message="A NumPy version.*")

    print("Reconstructing Images via MLP:")
    test_reconstruct_folder("MLP", folder_path, num_of_epochs=num_of_epochs)
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

    print("Reconstructing Images via SIREN_HYBRID")
    test_reconstruct_folder("SIREN_HYBRID", folder_path, num_of_epochs=num_of_epochs)
    print("")

    print("Comparing Reconstructions of SIREN_HYBRID:")
    test_compare_folder(folder_path, './Results/SIREN_HYBRID/')
    print("")

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


    print("Reconstructing Images via SIREN_DEEPER:")
    test_reconstruct_folder("SIREN_DEEPER", folder_path, num_of_epochs=num_of_epochs)
    print("")

    print("Comparing Reconstructions of SIREN_DEEPER:")
    test_compare_folder(folder_path, './Results/SIREN_DEEPER/')
    print("")

    print("Reconstructing Images via SIREN_SHALLOW:")
    test_reconstruct_folder("SIREN_SHALLOW", folder_path, num_of_epochs=num_of_epochs)
    print("")

    print("Comparing Reconstructions of SIREN_SHALLOW:")
    test_compare_folder(folder_path, './Results/SIREN_SHALLOW/')
    print("")

    print("Reconstructing Images via MLP_SINE")
    test_reconstruct_folder("MLP_SINE", folder_path, num_of_epochs=num_of_epochs)
    print("")

    print("Comparing Reconstructions of MLP_SINE:")
    test_compare_folder(folder_path, './Results/MLP_SINE/')
    print("")

    print("Reconstructing Images via MLP_SINE2")
    test_reconstruct_folder("MLP_SINE2", folder_path, num_of_epochs=num_of_epochs)
    print("")

    print("Comparing Reconstructions of MLP_SINE2:")
    test_compare_folder(folder_path, './Results/MLP_SINE2/')
    print("")

    print("Reconstructing Images via MLP_SINE3")
    test_reconstruct_folder("MLP_SINE3", folder_path, num_of_epochs=num_of_epochs)
    print("")

    print("Comparing Reconstructions of MLP_SINE3:")
    test_compare_folder(folder_path, './Results/MLP_SINE3/')
    print("")
