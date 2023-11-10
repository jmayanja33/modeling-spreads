import nfl_data_py as nfl
import requests
import os
import pandas as pd
from Simulator.divisions import team_abbreviations, team_markets

group_cols = ["recent_team", "season_type", "week"]

keep_cols = [
    "team_name", "playoff", "week", "completions", "attempts", "passing_yards", "passing_tds",
    "interceptions", "sacks", "sack_yards", "sack_fumbles", "sack_fumbles_lost", "passing_air_yards",
    "passing_yards_after_catch", "passing_first_downs", "passing_epa", "passing_2pt_conversions",
    "carries", "rushing_yards", "rushing_tds", "rushing_fumbles", "rushing_fumbles_lost",
    "rushing_first_downs", "rushing_epa", "rushing_2pt_conversions", "receiving_fumbles",
    "receiving_fumbles_lost", "special_teams_tds", "fantasy_points", "fantasy_points_ppr"
]


class NFLStats:
    """Class to extract nfl stats by week"""
    def __init__(self):
        self.data = None
        self.season = None
        self.api_key = os.getenv("SPORTS_DATA_IO_KEY")

    def update_season_data(self, season):
        """Function to get weekly data for a season"""
        df = nfl.import_weekly_data([season])
        df = df.groupby(group_cols, as_index=False).sum()
        df = df.rename(columns={"recent_team": "team_name", "season_type": "playoff"})
        self.data = df[keep_cols]
        self.season = season

    def format_playoff_data(self, week, data):
        """Function to format playoff data"""

        # Filter for playoff weeks after 2020 (17 game season)
        if self.season >= 2021:
            if week == "Wildcard":
                playoff_data = data[data["week"].isin(set([i for i in range(19)]))]
            elif week == "Division":
                playoff_data = data[data["week"].isin(set([i for i in range(20)]))]
            elif week == "Conference":
                playoff_data = data[data["week"].isin(set([i for i in range(21)]))]
            else:
                playoff_data = data
        # Filter for playoff weeks after 2020 (16 game season)
        else:
            if week == "Wildcard":
                playoff_data = data[data["week"].isin(set([i for i in range(18)]))]
            elif week == "Division":
                playoff_data = data[data["week"].isin(set([i for i in range(19)]))]
            elif week == "Conference":
                playoff_data = data[data["week"].isin(set([i for i in range(20)]))]
            else:
                playoff_data = data

        return playoff_data

    def extract_season_stats(self, team, week, playoff):
        """Function to extract stats weekly by team"""

        # Filter data
        if playoff == 1:
            filtered_data = self.data[(self.data["team_name"] == team)]
            filtered_data = self.format_playoff_data(week, filtered_data)

        else:
            filtered_data = self.data[self.data["team_name"] == team]
            filtered_data = filtered_data[filtered_data["week"] < int(week)]
            filtered_data = filtered_data[filtered_data["playoff"] == 'REG']

        # Compile statistics for all games before current week
        if len(filtered_data) > 0:
            completions = filtered_data["completions"].sum()
            attempts = filtered_data["attempts"].sum()
            passing_yards = filtered_data["passing_yards"].sum()
            passing_tds = filtered_data["passing_tds"].sum()
            interceptions = filtered_data["interceptions"].sum()
            sacks = filtered_data["sacks"].sum()
            sack_yards = filtered_data["sack_yards"].sum()
            sack_fumbles = filtered_data["sack_fumbles"].sum()
            sack_fumbles_lost = filtered_data["sack_fumbles_lost"].sum()
            passing_air_yards = filtered_data["passing_air_yards"].sum()
            passing_yards_after_catch = filtered_data["passing_yards_after_catch"].sum()
            passing_first_downs = filtered_data["passing_first_downs"].sum()
            passing_epa = filtered_data["passing_epa"].mean()
            passing_two_point_conversions = filtered_data["passing_2pt_conversions"].sum()
            carries = filtered_data["carries"].sum()
            rushing_yards = filtered_data["rushing_yards"].sum()
            rushing_tds = filtered_data["rushing_tds"].sum()
            rushing_fumbles = filtered_data["rushing_fumbles"].sum()
            rushing_fumbles_lost = filtered_data["rushing_fumbles_lost"].sum()
            rushing_first_downs = filtered_data["rushing_first_downs"].sum()
            rushing_epa = filtered_data["rushing_epa"].mean()
            rushing_two_point_conversions = filtered_data["rushing_2pt_conversions"].sum()
            receiving_fumbles = filtered_data["receiving_fumbles"].sum()
            receiving_fumbles_lost = filtered_data["receiving_fumbles_lost"].sum()
            special_teams_tds = filtered_data["special_teams_tds"].sum()
            fantasy_points = filtered_data["fantasy_points"].sum()
            fantasy_points_ppr = filtered_data["fantasy_points_ppr"].sum()

            extracted_data = [completions, attempts, passing_yards, passing_tds, interceptions, sacks, sack_yards,
                              sack_fumbles, sack_fumbles_lost, passing_air_yards, passing_yards_after_catch,
                              passing_first_downs, passing_epa, passing_two_point_conversions, carries, rushing_yards,
                              rushing_tds, rushing_fumbles, rushing_fumbles_lost, rushing_first_downs, rushing_epa,
                              rushing_two_point_conversions, receiving_fumbles, receiving_fumbles_lost,
                              special_teams_tds, fantasy_points, fantasy_points_ppr]

            return extracted_data

        else:
            return [0] * 27

    def find_standing_data(self, team_name):
        """Function to get a team's record/standing"""
        # Get standings data for 2023
        try:
            response = requests.get(f"https://api.sportsdata.io/v3/nfl/scores/json/Standings/2023?key={self.api_key}")
            standings = response.json()

            # Extract correct team data
            for team in standings:
                if team["Name"] == team_abbreviations[team_name]:
                    wins = team["Wins"]
                    losses = team["Losses"]
                    ties = team["Ties"]
                    place = team["DivisionRank"]
                    points_for = team["PointsFor"]
                    points_against = team["PointsAgainst"]
                    return [wins, losses, ties, place, points_for, points_against]

        except Exception as e:
            print(f"API call failed; Details: {e}")
            return [0, 0, 0, 0, 0, 0]

    def find_betting_data(self, team_name):
        """Function to get a team's record against the spread/point_total"""

        # Find spread data
        try:
            print(f"Finding record against the spread for team: {team_name}")
            spread_df = pd.read_html("https://betiq.teamrankings.com/nfl/betting-trends/spread-ats-records/")[0]
            team_spread_df = spread_df[spread_df["Team"] == team_markets[team_name]]
            team_spread_df.reset_index(inplace=True, drop=True)
            spread_record = team_spread_df["ATS Record"][0].split("-")
            spread_wins = spread_record[0] + spread_record[2]
            spread_losses = spread_record[1]

        except Exception as e:
            print(f"API call failed; Details: {e}")
            return [0, 0, 0]

        # Find over data
        try:
            print(f"Finding over point total record for team: {team_name}")
            point_total_df = pd.read_html("https://betiq.teamrankings.com/nfl/betting-trends/over-under-records/")[0]
            team_points_total_df = point_total_df[point_total_df["Team"] == team_markets[team_name]]
            team_points_total_df.reset_index(inplace=True, drop=True)
            over_record = team_points_total_df["Over Record"][0].split("-")
            over_wins = over_record[0] + over_record[2]
            over_losses = over_record[1]

        except Exception as e:
            print(f"Points Total Record API call failed; Details: {e}")
            return [0, 0, 0]

        return [spread_wins, spread_losses, over_wins, over_losses]

    def compile_all_stats(self, team_name, week, playoff):
        """Function to compile home team standings, betting, and statistical data"""
        standing_data = self.find_standing_data(team_name)
        betting_data = self.find_betting_data(team_name)
        statistical_data = self.extract_season_stats(team_name, week, playoff)

        compiled_data = standing_data + betting_data + statistical_data
        return compiled_data
