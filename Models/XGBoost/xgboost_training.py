"""
Run the main section at the bottom of this file to train and evaluate models for Spread Predictions,
Points total predictions, Favorite Cover Probabilities, and Over Cover Probabilities.

Inside the folder for each model in Models/XGBoost/F$dependent variable$ there will be
3 files. ScoreStats.txt has info on RMSE and R-squared (or Accuracy, Precision, Recall, and F1 for classification).
ScoreTrainingLoss.png contains a plot of the training losses to gauge over fitting. ScoreModel.keras is the model that
can be loaded at any time.

This script automatically tunes hyperparemeters for the final model through a grid search cross validation.
To edit which hyperparameters are tested and the range of values for each hyperparameter tested, edit the
`param_tuning` dictionary on lines 164-169. Grid search at the moment is set to n_jobs=-1, which uses all available
cores on the machine. To change this, edit n_jobs in the `params_model` object (lines 176 and 180)
to the maximum desired number of cores.

"""

import pickle
import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
from numpy import sort
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from Models.model_evaluator import *
from root_path import ROOT_PATH

# Ignore sklearn warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

dependent_variables = {
    "Spread": "actual_spread",
    "Favorite Cover Probability": "favorite_covered",
    "Over Cover Probability": "over_covered",
    "Points Total": "total_points"
}

drop_cols = ([dependent_variables[i] for i in dependent_variables.keys()] +
             ["level_0", "home_team_name", "home_team_division_name", "away_team_name", "away_team_division_name",
              "stadium_name", "kickoff_temperature", "kickoff_wind", "kickoff_humidity", "home_score", "away_score"])


class XGBoostTrainer:
    """Class to train an xgboost model"""

    def __init__(self, dependent_variable):
        self.dependent_variable = dependent_variable
        self.significant_feature_names = None
        self.threshold = None
        self.best_params = None
        self.model_folder_path = create_model_folder("XGBoost", self.dependent_variable)

        # Load data
        print("Loading data")
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

        # TODO: Fill missing temperature values from pro weather api

    def evaluate_threshold(self, thresh, model_thresholds, num_thresholds, predictions, file_name, counter, classification=False):
        """Function to evaluate a model threshold"""
        if not classification:
            rmse = mean_squared_error(self.y_validation, predictions, squared=False)
            model_thresholds[thresh] = rmse
            # Save to file
            file_name.write(f"\n- Threshold: {thresh}  - RMSE: {rmse}")
            print(f"Evaluating feature threshold: {thresh}; - RMSE: {rmse}; - PROGRESS: {counter}/{num_thresholds}")

        # Accuracy for bet win prediction/over prediction
        else:
            accuracy = accuracy_score(self.y_validation, predictions)
            model_thresholds[thresh] = accuracy
            # Save to file
            file_name.write(f"\n- Threshold: {thresh}  - Accuracy: {accuracy}")
            print(f"Evaluating feature threshold: {thresh}; - Accuracy: {accuracy}; - PROGRESS: {counter}/{num_thresholds}")

    def find_best_threshold(self, model_thresholds, classification):
        """Function to detect best model threshold"""
        if classification:
            # Find threshold that has best accuracy
            max_accuracy = max(model_thresholds.values())
            for thresh in model_thresholds:
                if model_thresholds[thresh] == max_accuracy:
                    self.threshold = thresh
                    break
            print(f"Found best threshold is {self.threshold} with Accuracy of {max_accuracy}")
        else:
            # Find threshold that has best RMSE
            min_rmse = min(model_thresholds.values())
            for thresh in model_thresholds:
                if model_thresholds[thresh] == min_rmse:
                    self.threshold = thresh
                    break
            print(f"Found best threshold is {self.threshold} with RMSE of {min_rmse}")

    def find_feature_importance(self):
        """Function to plot and select most important features using the elbow method"""
        print("Training XGB model with default hyperparameters to find feature importance")

        # Find feature significance
        if self.dependent_variable in {'Spread', 'Points Total'}:
            model = XGBRegressor(random_state=33)
        else:
            model = XGBClassifier(random_state=33)
        model.fit(self.X_train, self.y_train)

        print("Finding feature significance")
        thresholds = sort(list(model.feature_importances_))

        # Create a model using each feature importance as a feature threshold, pick threshold with lowest RMSE
        model_thresholds = dict()
        file = open(f"{self.model_folder_path}/{self.dependent_variable.replace(' ', '')}ThresholdEval.txt", 'w')
        counter = 1
        num_thresholds = len(thresholds)
        # Iterate through thresholds
        for thresh in thresholds:
            features = SelectFromModel(estimator=model, threshold=thresh, prefit=True)
            # Transform feature sets
            features_X_train = features.transform(self.X_train)
            features_X_validation = features.transform(self.X_validation)

            # Fit model
            if self.dependent_variable in {'Spread', 'Points Total'}:
                feature_model = XGBRegressor(random_state=33)
            else:
                feature_model = XGBClassifier(random_state=33)
            feature_model.fit(features_X_train, self.y_train)

            # Make predictions
            predictions = feature_model.predict(features_X_validation)

            # RMSE for spread (regression
            if self.dependent_variable in {'Spread', 'Points Total'}:
                self.evaluate_threshold(thresh, model_thresholds, num_thresholds, predictions, file, counter,
                                        classification=False)
                counter += 1

            # Accuracy for bet win prediction/over prediction
            else:
                self.evaluate_threshold(thresh, model_thresholds, num_thresholds, predictions, file, counter,
                                        classification=True)
                counter += 1

        # Find threshold that has best accuracy metric
        if self.dependent_variable in {'Spread', 'Points Total'}:
            self.find_best_threshold(model_thresholds, classification=False)
        else:
            self.find_best_threshold(model_thresholds, classification=True)

        print("Filtering data to only include selected features")

        # Use the best threshold for final feature selection
        significant_features = SelectFromModel(model, threshold=self.threshold, prefit=True)
        self.significant_feature_names = [self.X_train.columns[i] for i in significant_features.get_support(indices=True)]

        # Save most significant features to a file
        print("Writing feature names and importance to a file")
        if not os.path.exists(f"{ROOT_PATH}/Models/XGBoost/{self.dependent_variable.replace(' ', '')}/FeatureSelection"):
            os.mkdir(f"{ROOT_PATH}/Models/XGBoost/{self.dependent_variable.replace(' ', '')}/FeatureSelection")

        with open(f"{ROOT_PATH}/Models/XGBoost/{self.dependent_variable.replace(' ', '')}/FeatureSelection/MostSignificantFeatures.pkl", "wb") as pklfile:
            pickle.dump(self.significant_feature_names, pklfile)
            pklfile.close()

        # Save second file with importance
        importance_vals = model.feature_importances_
        importance_dict = dict(sorted({model.feature_names_in_[i]: str(importance_vals[i]) for i in range(len(importance_vals)) if importance_vals[i] >= self.threshold}.items(),
                                      key=lambda x: x[1], reverse=True))
        with open(f"{ROOT_PATH}/Models/XGBoost/{self.dependent_variable.replace(' ', '')}/FeatureSelection//MostSignificantFeatureValues.txt", "w") as file:
            file.write(str(importance_dict))
            file.close()

        # Update X_train and X_test with selected features
        self.X_train = significant_features.transform(self.X_train)
        self.X_validation = significant_features.transform(self.X_validation)
        self.X_test = significant_features.transform(self.X_test)

    def find_optimal_hyperparameters(self):
        """Function which uses cross validation to find optimal model hyperparameters"""
        # Create xgboost D-matrices
        print("Creating D-matrices and setting parameter values for cross validation")
        d_train = xgb.DMatrix(self.X_train, self.y_train, enable_categorical=True)
        d_test = xgb.DMatrix(self.X_validation, self.y_validation, enable_categorical=True)

        # Create dictionary of potential parameters for testing in cross validation
        param_tuning = {
            "max_depth": np.arange(3, 10),
            "learning_rate": np.arange(0.1, 1, 0.1),
            "n_estimators": np.arange(100, 1000, 100),
            "gamma": np.arange(0, 5)
        }

        # Use grid search to perform k-fold cross validation with k=5 to find best parameters
        print("Performing 5 fold cross validation:")
        # Create xgb object
        if self.dependent_variable in {'Spread', 'Points Total'}:
            xgb_object = XGBRegressor(random_state=33)
            params_model = GridSearchCV(estimator=xgb_object, param_grid=param_tuning, scoring="neg_mean_squared_error",
                                        verbose=10, n_jobs=-1)
        else:
            xgb_object = XGBClassifier(random_state=33)
            params_model = GridSearchCV(estimator=xgb_object, param_grid=param_tuning, scoring="accuracy",
                                        verbose=10, n_jobs=-1)

        # Find best params
        params_model.fit(self.X_train, self.y_train)
        self.best_params = params_model.best_params_

    def train_model(self):
        """Function to train an xgboost model"""
        print(f"Creating final {self.dependent_variable} model with best parameters from cross validation")

        # Spread model
        if self.dependent_variable in {'Spread', 'Points Total'}:
            model = XGBRegressor(**self.best_params, random_state=33)
            model.fit(self.X_train, self.y_train)

            # Evaluate model
            print(f"Collecting model {self.dependent_variable} performance statistics")
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
        # Bet win probability model
        else:
            model = XGBClassifier(**self.best_params, random_state=33)
            model.fit(self.X_train, self.y_train)

            # Evaluate model
            print(f"Collecting model {self.dependent_variable} performance statistics")
            calculate_performance_metrics_classification('NeuralNetwork',
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

        # Save Model
        print(f"Saving {self.dependent_variable} model and performance statistics")
        model.save_model(f"{self.model_folder_path}/{self.dependent_variable.replace(' ', '')}XGBoostModel.json")

    def create_model(self):
        self.find_feature_importance()
        self.find_optimal_hyperparameters()
        self.train_model()


if __name__ == '__main__':
    #  Initialize model trainers
    spread_trainer = XGBoostTrainer("Spread")
    favorite_cover_trainer = XGBoostTrainer("Favorite Cover Probability")
    over_trainer = XGBoostTrainer("Over Cover Probability")
    points_total_trainer = XGBoostTrainer("Points Total")

    # Train BigModels
    # spread_trainer.create_model()
    favorite_cover_trainer.create_model()
    over_trainer.create_model()
    points_total_trainer.create_model()


