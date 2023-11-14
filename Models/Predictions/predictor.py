from Scrapers.nfl_stats_api import NFLStats
from Simulator.stadiums import stadiums
from Simulator.divisions import teams, divisions, division_ids


class Predictor:

    def __init__(self, week, home_team, away_team, favorite, given_spread, given_total, stadium, playoff=0,
                 neutral_site=0):
        self.model = None

        self.nfl = NFLStats()
        self.nfl.update_season_data(season=2023)

        self.data = self.nfl.collect_data_for_predictions(week, home_team, away_team, favorite, given_spread,
                                                          given_total, stadium, playoff, neutral_site)

    def predict_spread(self):
        """Function to predict spread"""
        raise NotImplementedError

    def predict_favorite_to_cover(self):
        """Function to predict if favorite covers"""
        raise NotImplementedError

    def predict_total_points(self):
        """Function to predict total points"""
        raise NotImplementedError

    def predict_over_to_cover(self):
        """Function to predict if over covers"""
        raise NotImplementedError
