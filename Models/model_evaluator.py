"""
Helper script with functions to evaluate a model's performance and save accuracy metrics to a file.
"""
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import os
from root_path import ROOT_PATH


def create_model_folder(model_type, dependent_variable):
    """Function to ensure folder to save model metrics exists"""
    model_folder_path = f"{ROOT_PATH}/Models/{model_type}"
    dependent_variable_path = f"{model_folder_path}/{dependent_variable.replace(' ', '')}"
    if not os.path.exists(model_folder_path):
        os.mkdir(model_folder_path)
    if not os.path.exists(dependent_variable_path):
        os.mkdir(dependent_variable_path)

    return dependent_variable_path


def round_predictions(predictions):
    """Functions to round softmax predictions for evaluation"""
    rounded_predictions = []
    for i in predictions:
        if i[0] > i[1]:
            rounded_predictions.append(0)
        else:
            rounded_predictions.append(1)
    return rounded_predictions


def calculate_adj_r2(r2, data):
    """
    Function to calculate adjusted r2
    :param r2: Previously calculated regular r2 number
    :param data: Dataframe from which r^2 was calculated
    :return:  Adjusted r2 value
    """
    df = pd.DataFrame(data)
    num_observations = len(df)
    num_features = len(df.columns)
    return 1 - (1-r2) * (num_observations-1)/(num_observations-num_features-1)


def calculate_performance_metrics_regression(model_type, dependent_variable, model, X_train, X_validation, X_test, y_train,
                                             y_validation, y_test, best_params=None, significant_feature_names=None):
    """
     Function to calculate RMSE, R-squared, and Adj. R-squared, and save to a file
    :param model_type: The kind of model being evaluated, one of 'XGBoost' or 'NeuralNetwork'
    :param dependent_variable:  A string, either 'Spread', 'Over Probability', or 'Favorite Cover Probability'
    :param model: The model being evaluated
    :param X_train: Training set features
    :param X_validation: Validation set features
    :param X_test: Test set features
    :param y_train: Training set actual values
    :param y_validation: Validation set actual values
    :param y_test: Test set actual values
    :param best_params: Best hyperparameters (xgboost only)
    :param significant_feature_names: Features used if feature selection occurred
    :return: None
    """

    # Make predictions
    train_predictions = model.predict(X_train)
    validation_predictions = model.predict(X_validation)
    test_predictions = model.predict(X_test)

    # Calculate RMSE
    train_rmse = mean_squared_error(y_train, train_predictions, squared=False)
    validation_rmse = mean_squared_error(y_validation, validation_predictions, squared=False)
    test_rmse = mean_squared_error(y_test, test_predictions, squared=False)

    # Calculate R-squared
    train_r2 = r2_score(y_train, train_predictions)
    validation_r2 = r2_score(y_validation, validation_predictions)
    test_r2 = r2_score(y_test, test_predictions)

    # Calculate Adj. R-Squared
    train_adj_r2 = calculate_adj_r2(train_r2, X_train)
    validation_adj_r2 = calculate_adj_r2(validation_r2, X_validation)
    test_adj_r2 = calculate_adj_r2(test_r2, X_test)

    # Write evaluation stats to a file
    folder_path = create_model_folder(model_type, dependent_variable)
    file = open(f"{folder_path}/ModelEvaluationStats.txt", 'w')
    file.write(f"""\n- Selected Features: {significant_feature_names}
        \n- Best Params: {best_params}
        \n- Training RMSE: {train_rmse}\n- Training R-Squared: {train_r2}\n- Training Adj. R-squared: {train_adj_r2}
        \n- Validation RMSE: {validation_rmse}\n- Test R-squared: {validation_r2}\n- Test Adj R-Squared: {validation_adj_r2}
        \n- Test RMSE: {test_rmse}\n- Test R-squared: {test_r2}\n- Test Adj R-Squared: {test_adj_r2}
        """)
    file.close()


def calculate_performance_metrics_classification(model_type, dependent_variable, model, X_train, X_validation, X_test,
                                                 y_train, y_validation, y_test, best_params=None,
                                                 significant_feature_names=None):
    """
     Function to calculate RMSE, R-squared, and Adj. R-squared, and save to a file
    :param model_type: The kind of model being evaluated, one of 'XGBoost' or 'NeuralNetwork'
    :param dependent_variable: A string, either 'Spread', 'Over Probability', or 'Favorite Cover Probability'
    :param model: The model being evaluated
    :param X_train: Training set features
    :param X_validation: Validation set features
    :param X_test: Test set features
    :param y_train: Training set actual values
    :param y_validation: Validation set actual values
    :param y_test: Test set actual values
    :param best_params: Best hyperparameters (xgboost only)
    :param significant_feature_names: Features used if feature selection occurred
    :return: None
    """

    # Make predictions
    train_predictions = model.predict(X_train).round(decimals=0)
    validation_predictions = model.predict(X_validation).round(decimals=0)
    test_predictions = model.predict(X_test).round(decimals=0)

    train_predictions = round_predictions(train_predictions)
    validation_predictions = round_predictions(validation_predictions)
    test_predictions = round_predictions(test_predictions)

    # Calculate Accuracy
    train_accuracy = accuracy_score(y_train, train_predictions)
    validation_accuracy = accuracy_score(y_validation, validation_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    # Calculate Precision
    train_precision = precision_score(y_train, train_predictions)
    validation_precision = precision_score(y_validation, validation_predictions)
    test_precision = precision_score(y_test, test_predictions)

    # Calculate Recall
    train_recall = recall_score(y_train, train_predictions)
    validation_recall = recall_score(y_validation, validation_predictions)
    test_recall = recall_score(y_test, test_predictions)

    # Calculate F1 Score
    train_f1 = f1_score(y_train, train_predictions)
    validation_f1 = f1_score(y_validation, validation_predictions)
    test_f1 = f1_score(y_test, test_predictions)

    # Write evaluation stats to a file
    folder_path = create_model_folder(model_type, dependent_variable)
    file = open(f"{folder_path}/ModelEvaluationStats.txt", 'w')
    file.write(f"""\n- Selected Features: {significant_feature_names}
        \n- Best Params: {best_params}
        \n- Training Accuracy: {train_accuracy}\n- Training Precision: {train_precision}\n- Training Recall: {train_recall}\n- Training F1 Score: {train_f1}
        \n- Validation Accuracy: {validation_accuracy}\n- Validation Precision: {validation_precision}\n- Validation Recall: {validation_recall}\n- Validation F1 Score: {validation_f1}
        \n- Test Accuracy: {test_accuracy}\n- Test Precision: {test_precision}\n- Test Recall: {test_recall}\n- Test F1 Score: {test_f1}
        """)
    file.close()
