from Scrapers.nfl_statistics_api import NFLStatsAPI
from tensorflow import keras


class Predictor:

    def __init__(self, week, home_team, away_team, favorite, given_spread, given_total, stadium, playoff=0,
                 neutral_site=0):
        self.nfl = NFLStatsAPI()
        self.nfl.update_season_data(season=2023)

        self.data = self.nfl.collect_data_for_predictions(week, home_team, away_team, favorite, given_spread,
                                                          given_total, stadium, playoff, neutral_site)

    def predict_spread(self):
        """Function to predict spread"""
        model = keras.models.load_model("Models/NeuralNetwork/Spread/SpreadSavedModel")
        prediction = round(model.predict(self.data)[0][0], 0)
        return float(prediction)

    def predict_favorite_to_cover(self):
        """Function to predict if favorite covers"""
        model = keras.models.load_model("Models/NeuralNetwork/FavoriteCoverProbability/FavoriteCoverProbabilitySavedModel")
        prediction = round(model.predict(self.data)[0][0], 4)
        return float(prediction)

    def predict_total_points(self):
        """Function to predict total points"""
        model = keras.models.load_model("Models/NeuralNetwork/PointsTotal/PointsTotalSavedModel")
        prediction = round(model.predict(self.data)[0][0], 0)
        return float(prediction)

    def predict_over_to_cover(self):
        """Function to predict if over covers"""
        model = keras.models.load_model("Models/NeuralNetwork/OverCoverProbability/OverCoverProbabilitySavedModel")
        prediction = round(model.predict(self.data)[0][0], 4)
        return float(prediction)
