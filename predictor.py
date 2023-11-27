from Scrapers.nfl_statistics_api import NFLStatsAPI
from tensorflow import keras
from xgboost import XGBRegressor
from root_path import ROOT_PATH
import pandas as pd
import pickle


def rename_columns(df, reverse=False):
    """Function to rename columns to match model data frame"""
    columns = df.columns
    column_names = dict()
    for i in range(len(columns)):
        column = columns[i]
        column_names[column] = i
    return df.rename(columns=column_names)


class Predictor:

    def __init__(self, week, home_team, away_team, favorite, given_spread, given_total, stadium, playoff=0,
                 neutral_site=0):
        self.nfl = NFLStatsAPI()
        self.nfl.update_season_data(season=2023)

        self.data = self.nfl.collect_data_for_predictions(week, home_team, away_team, favorite, given_spread,
                                                          given_total, stadium, playoff, neutral_site)

    def predict_spread(self):
        """Function to predict spread"""
        with open(f"Models/XGBoost/Spread/FeatureSelection/MostSignificantFeatures.pkl", "rb") as pklfile:
            significant_features = pickle.load(pklfile)
            pklfile.close()

        formatted_data = rename_columns(self.data[significant_features])

        model = XGBRegressor()
        model.load_model(f"Models/XGBoost/Spread/SpreadXGBoostModel.json")
        prediction = round(model.predict(formatted_data)[0], 0)


        # model = keras.models.load_model("Models/NeuralNetwork/Spread/SpreadSavedModel")
        # prediction = round(model.predict(self.data)[0][0], 0)
        return float(prediction)

    def predict_favorite_to_cover(self):
        """Function to predict if favorite covers"""
        with open(f"Models/XGBoost/FavoriteCoverProbability/FeatureSelection/MostSignificantFeatures.pkl", "rb") as pklfile:
            significant_features = pickle.load(pklfile)
            pklfile.close()

        formatted_data = rename_columns(self.data[significant_features])

        model = XGBRegressor()
        model.load_model(f"Models/XGBoost/FavoriteCoverProbability/FavoriteCoverProbabilityXGBoostModel.json")
        raw_prediction = model.predict(formatted_data)[0]

        if raw_prediction[0] > raw_prediction[1]:
            return 0, round(float(raw_prediction[0]), 4)
        else:
            return 1, round(float(raw_prediction[1]), 4)

        # prediction = round(max(raw_prediction[0], raw_prediction[1]), 4)

        # model = keras.models.load_model("Models/NeuralNetwork/FavoriteCoverProbability/FavoriteCoverProbabilitySavedModel")
        # prediction = round(model.predict(self.data)[0][0], 4)
        # return float(prediction)

    def predict_total_points(self):
        """Function to predict total points"""
        with open(f"Models/XGBoost/PointsTotal/FeatureSelection/MostSignificantFeatures.pkl", "rb") as pklfile:
            significant_features = pickle.load(pklfile)
            pklfile.close()

        formatted_data = rename_columns(self.data[significant_features])

        model = XGBRegressor()
        model.load_model(f"Models/XGBoost/PointsTotal/PointsTotalXGBoostModel.json")
        prediction = round(model.predict(formatted_data)[0], 0)

        # model = keras.models.load_model("Models/NeuralNetwork/PointsTotal/PointsTotalSavedModel")
        # prediction = round(model.predict(self.data)[0][0], 0)
        return float(prediction)

    def predict_over_to_cover(self):
        """Function to predict if over covers"""
        with open(f"Models/XGBoost/OverCoverProbability/FeatureSelection/MostSignificantFeatures.pkl", "rb") as pklfile:
            significant_features = pickle.load(pklfile)
            pklfile.close()

        formatted_data = rename_columns(self.data[significant_features])

        model = XGBRegressor()
        model.load_model(f"Models/XGBoost/OverCoverProbability/OverCoverProbabilityXGBoostModel.json")
        raw_prediction = model.predict(formatted_data)[0]
        if raw_prediction[0] > raw_prediction[1]:
            return 0, round(float(raw_prediction[0]), 4)
        else:
            return 1, round(float(raw_prediction[1]), 4)
        # prediction = round(max(raw_prediction[0], raw_prediction[1]), 4)

        # model = keras.models.load_model("Models/NeuralNetwork/OverCoverProbability/OverCoverProbabilitySavedModel")
        # prediction = round(model.predict(self.data)[0][0], 4)
        # return float(prediction)
