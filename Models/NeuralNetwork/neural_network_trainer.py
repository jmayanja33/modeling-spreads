"""
Run the main section at the bottom of this file to train and evaluate models for Spread Predictions,
Points total predictions, Favorite Cover Probabilities, and Over Cover Probabilities.

Inside the folder for each model in Models/NeuralNetwork/NeuralNetwork/$dependent variable$ there will be 3 files.
ScoreStats.txt has info on RMSE and R-squared, (or Accuracy, Precision, Recall, and F1 for classification).
ScoreTrainingLoss.png contains a plot of the training losses to gauge over fitting.
ScoreModel.keras is the model that can be loaded at any time.

To edit the number of hidden layers/neurons per layer for the models, edit lines 60-109.
"""


import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from Models.model_evaluator import *
from root_path import ROOT_PATH
import os
import shutil

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set seed
keras.utils.set_random_seed(33)

dependent_variables = {
    "Spread": "actual_spread",
    "Favorite Cover Probability": "favorite_covered",
    "Over Cover Probability": "over_covered",
    "Points Total": "total_points"
}

drop_cols = ([dependent_variables[i] for i in dependent_variables.keys()] +
             ["level_0", "home_team_name", "home_team_division_name", "away_team_name", "away_team_division_name",
              "stadium_name", "kickoff_temperature", "kickoff_wind", "kickoff_humidity", "home_score", "away_score"])


class NeuralNetworkTrainer:
    """Class to train a neural network"""

    def __init__(self, dependent_variable):
        self.dependent_variable = dependent_variable
        self.best_params = None
        self.model_folder_path = create_model_folder("NeuralNetwork", self.dependent_variable)

        print("Loading data")
        self.data = pd.read_csv(f"{ROOT_PATH}/Data/expanded_data.csv")
        self.training_set = pd.read_csv(f"{ROOT_PATH}/Data/SplitData/TrainingSet.csv")
        self.validation_set = pd.read_csv(f"{ROOT_PATH}/Data/SplitData/ValidationSet.csv")
        self.test_set = pd.read_csv(f"{ROOT_PATH}/Data/SplitData/TestSet.csv")
        self.X_train = self.training_set.drop(drop_cols, axis=1)
        self.X_validation = self.validation_set.drop(drop_cols, axis=1)
        self.X_test = self.test_set.drop(drop_cols, axis=1)
        self.y_train = self.training_set[dependent_variables[dependent_variable]]
        self.y_validation = self.validation_set[dependent_variables[dependent_variable]]
        self.y_test = self.test_set[dependent_variables[dependent_variable]]

        # Format columns
        self.X_train["week"] = self.X_train["week"].replace(
            {"Wildcard": 101, "Division": 102, "Conference": 103, "Superbowl": 104}
        )
        self.X_validation["week"] = self.X_validation["week"].replace(
            {"Wildcard": 101, "Division": 102, "Conference": 103, "Superbowl": 104}
        )
        self.X_test["week"] = self.X_test["week"].replace(
            {"Wildcard": 101, "Division": 102, "Conference": 103, "Superbowl": 104}
        )

        self.X_train["week"] = pd.to_numeric(self.X_train["week"])
        self.X_validation["week"] = pd.to_numeric(self.X_validation["week"])
        self.X_test["week"] = pd.to_numeric(self.X_test["week"])

    def train_model(self):
        """Function to train model"""
        print("Training neural network")

        # Spread model hyperparameters
        if self.dependent_variable == 'Spread':
            model = keras.Sequential([
                keras.layers.InputLayer(input_shape=(94,)),
                keras.layers.Dense(94, activation='relu', name='hidden1'),
                keras.layers.Dense(94, activation='relu', name='hidden2'),
                keras.layers.Dense(94, activation='relu', name='hidden3'),
                keras.layers.Dense(1, name='output', activation='linear')
            ])
            model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

        # Spread model hyperparameters
        elif self.dependent_variable == 'Points Total':
            model = keras.Sequential([
                keras.layers.InputLayer(input_shape=(94,)),
                keras.layers.Dense(94, activation='relu', name='hidden1'),
                keras.layers.Dense(94, activation='relu', name='hidden2'),
                keras.layers.Dense(94, activation='relu', name='hidden3'),
                keras.layers.Dense(1, name='output', activation='linear')
            ])
            model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

        # Over model hyperparameters
        elif self.dependent_variable == 'Over Probability':
            model = keras.Sequential([
                keras.layers.InputLayer(input_shape=(94,)),
                keras.layers.Dense(94, activation='relu', name='hidden1'),
                keras.layers.Dense(94, activation='relu', name='hidden2'),
                keras.layers.Dense(94, activation='relu', name='hidden3'),
                keras.layers.Dense(1, name='output', activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Favorite cover spread model hyperparameters
        else:
            model = keras.Sequential([
                keras.layers.InputLayer(input_shape=(94,)),
                keras.layers.Dense(94, activation='relu', name='hidden1'),
                keras.layers.Dense(94, activation='relu', name='hidden2'),
                keras.layers.Dense(94, activation='relu', name='hidden3'),
                keras.layers.Dense(1, name='output', activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Fit model
        early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=5)
        model.fit(self.X_train, self.y_train, epochs=2000, batch_size=2048, verbose=1,
                  validation_data=(self.X_validation, self.y_validation), callbacks=[early_stop])

        # Plot training losses
        print("Plotting training losses")
        history = pd.DataFrame(model.history.history)
        plt.figure(figsize=(6, 4))
        plt.plot(history["loss"], label="Loss")
        plt.plot(history["val_loss"], label="Val. Loss")
        plt.title(f"{self.dependent_variable} Training Loss")
        plt.legend()
        plt.savefig(f"{self.model_folder_path}/TrainingLoss.png")

        # Plot training accuracy
        if self.dependent_variable not in {'Spread', 'Points Total'}:
            plt.figure(figsize=(6, 4))
            plt.plot(history["accuracy"], label="Accuracy")
            plt.plot(history["val_accuracy"], label="Val. Accuracy")
            plt.title(f"{self.dependent_variable} Training Accuracy")
            plt.legend()
            plt.savefig(f"{self.model_folder_path}/TrainingAccuracy.png")

        # Evaluate model
        print(f"Collecting {self.dependent_variable} model performance statistics")
        if self.dependent_variable in {'Spread', 'Points Total'}:
            calculate_performance_metrics_regression('NeuralNetwork',
                                                     self.dependent_variable,
                                                     model,
                                                     self.X_train,
                                                     self.X_validation,
                                                     self.X_test,
                                                     self.y_train,
                                                     self.y_validation,
                                                     self.y_test,
                                                     best_params=self.best_params
                                                     )
        else:
            calculate_performance_metrics_classification('NeuralNetwork',
                                                         self.dependent_variable,
                                                         model,
                                                         self.X_train,
                                                         self.X_validation,
                                                         self.X_test,
                                                         self.y_train.tolist(),
                                                         self.y_validation.tolist(),
                                                         self.y_test.tolist(),
                                                         best_params=self.best_params
                                                         )

        # Save model
        print(f"Saving {self.dependent_variable} model and performance statistics")
        tf.keras.models.save_model(model, f"{self.model_folder_path}/{self.dependent_variable.replace(' ', '')}SavedModel")


# Run Script
if __name__ == '__main__':
    #  Initialize model trainers
    spread_trainer = NeuralNetworkTrainer("Spread")
    favorite_cover_trainer = NeuralNetworkTrainer("Favorite Cover Probability")
    over_trainer = NeuralNetworkTrainer("Over Cover Probability")
    total_trainer = NeuralNetworkTrainer("Points Total")

    # Train BigModels
    spread_trainer.train_model()
    favorite_cover_trainer.train_model()
    over_trainer.train_model()
    total_trainer.train_model()
