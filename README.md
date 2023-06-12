# Versatile Hybrid Recommender System

Welcome to the repository of the project "Developing a Versatile Hybrid Recommender System: Case Study on Programming Task Prediction". This project focuses on the development of a hybrid recommender system for predicting suitable programming tasks. The proposed tool is a versatile solution, applicable not only to programming task prediction but to other predictive tasks as well. 


## Table of Contents

- [Prerequisites](#prerequisites)
- [Structure](#structure)
- [Execution](#execution)
- [Data](#data)
- [Author](#author)
- [Acknowledgments](#acknowledgments)

## Prerequisites

Following are the main packages you need to install to run the project along with their versions:
- pandas 1.4.4
- numpy 1.21.5
- scikit-learn 1.0.2
- future 0.18.2
There is also a requirements file that cand be used for the installation with the following command:
```bash
pip install -r requirements.txt
```
## Structure
- data (folder)
- main.py
- data_creation.py
- methods_pipeline.py
- system_architecture.py
- delivery_ui.py
- tests.py
- requirements.txt
- README.md

## Execution

Once you have cloned the repository and installed the prerequisites, navigate to the project's root directory. In the project's root directory, you will find a Python file named 'main.py'. Run this file, in order to start the tool.

The tool will prompt you for input during execution. Follow the on-screen prompts to input the required data.

First, choose if you want to use the tool for predictions (choose "Programming task") or for testing the tool.

For testing the system, two options are available: using programming task prediction data or using other datasets (new topic). 
- For using the tool to predict programming tasks, two CSV files are preloaded in the correct file location. 
- For using other datasets, a preloaded diabetes prediction dataset with 4000 entries is included.

After selecting the purpose and dataset, users are prompted to choose from eight transformation options. You can select any number of transformations according to your needs. 

The next pop-up confirms with the users the selected transformations. Here you have the choice of going back and reselecting.

In the case of testing the system, users will be asked if they want to retrain the models. This is recommended if you are using new datasets. Otherwise, you can select 'NO'.

Finally, if any of the transformations include a confidence interval or a threshold value, users are asked if they want to modify the default values of 0.2 and 0.5. Here you can change it to something else (between 0 and 0.5 for confidence intervals and between 0 and 1 for thresholds) or even add multiple confidence intervals or thresholds. The new values are then confirmed with the users.

After this last step, the tool should begin testing or predicting.

For testing, a shortened variant of the results will pop up once the tool is finished, while for the prediction, the full answers will appear. The full results can also be found in the corresponding data folder.

For testing with a new dataset and for predicting, the data used can be changed to different datasets inputed by the users as long as these datasets are saved accordingly, as described in the Data section.
## Data

The structure of the folder inside the data folder is as follows:
- data_final
- data_initial
    - course
    - Kaggle
- predictions
    - transformation
- results
    - loop
    - original
- test_data
    - data
    - results
        - original
        - predictions
            - transformation
- tool
    - new_topic
        - results
            - predictions
                - transformation
    - programming_prediction
        - load_test
        - results_predicting
            - predictions
                - transformation
        - results_testing
            - predictions
                - transformation

For the purpose of the project, the results presented in the paper are based on the data in predictions and results. The data_initial contains the initial 25 CSV files that were used for setting up the data. The CSV files in data_final are the ones being used when data is loaded at any point in the project. When specific functions are run without loading the data from data_final, files 'characteristics_1000_temp.csv' and 'mapping_1000_temp.csv' appear in this folder. The test_data folder contains the data necessary for the test file to run smoothly, while the results folder gets updated every time the test file is run. The original folder contains data that is necessary for running the test file. 

The folder tool contains the data of interest for running the tool.

File 'latest_results.txt' gets updated with results every time the tool is run.

Folder new_topic is reserved for running testing with a new topic. If a new dataset is wished then in this folder, 'full_characteristics.csv' and 'full_mapping.csv' need to be switched with files with the same names but containing new data. The first file needs to contain the predictors while the second one needs to contain the target values. If the files are changed, it is mandatory to do retraining as the tool will error otherwise.

Folder programming_prediction contains the data for when the topic is that of predicting suitable programming tasks. The two CSV files inside are necessary both for predicting and testing. Folder load_test is reserved for predicting. This is the folder where the user can place any CSV file for which they want to receive suitable programming task predictions. They simply need to place them here as CSV files. Folders results_predicting and results_testing are folders with similar structures and are where the resulting data is stored for those specific runs of the tool.

## Author

**Maria Emanuela Dumitru Toader**

## Acknowledgments

The code was created as part of the Final Bachelor Project and is submitted in partial fulfillment of the requirements of the degree of Joint Bachelor of Science in Data Science from the Eindhoven University of Technology and Tilburg University under the supervision of Prof.Dr. M. van den Brand, Dr. M. Seraj, and Dr. L. Ochoa Venegas.
