Computer Vision Lab                  Project
Dvir Ben Asuli                       318208816
The Hebrew University of Jerusalem   2024

Files:
    1. main.py- Functions for re-constructing a single image,
                via input from the user: Model name and Image name.
                Assumes that an image of the name provided by user exists in directory.
                Available Models: MLP, SIREN, SIREN_HYBRID, SIREN_NARROW, SIREN_WIDER,
                                  SIREN_DEEPER, SIREN_SHALLOW, MLP_SINE, MLP_SINE2, MLP_SINE3.
    2. common.py- Common functions for all tests and all models.
    3. models.py- Implementation og all models.
    4. test*.py- The 7 different tests, testing all available models.
       test_all.py- All tests consequently, including comparisons.

Folders:
    1. Images: Source images.
    2. Results: Result images of running all 7 tests, and their training loss graphs.
    3. Other: Diagrams of models and results of a specific exam.


Run preferably on Python3+.
    In order to run one of the tests just run python + the filename of the desired test.
    In order to test a specific image on a specific model, run python + main.py,
      afterwards enter image name and then the model name.

NOTE: Images in source "/Images" folder needs to be in ".jpg" format.
      Results will be generated in ".png" format inside "/Results" directory, according to the chosen model.
