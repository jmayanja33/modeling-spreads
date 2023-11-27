import nfl_data_py as nfl
import requests
import os
import pandas as pd
from Simulator.stadiums import stadiums
from Simulator.divisions import division_ids, divisions, teams, team_abbreviations, team_markets
from dotenv import load_dotenv

columns = [
    'year', 'week', 'playoff_game', 'home_team_id',
    'home_team_division_id', 'away_team_id',
    'away_team_division_id', 'favorite', 'given_spread', 'given_total',
    'stadium_id', 'stadium_city', 'stadium_open_date', 'stadium_roof_type', 'stadium_weather_type',
    'stadium_capacity', 'stadium_surface', 'stadium_latitude', 'stadium_longitude', 'neutral_site',
    'home_team_wins', 'home_team_losses', 'home_team_tie',
    'home_team_division_place', 'home_team_points_for', 'home_team_points_against', 'home_team_cover',
    'home_team_fail_cover', 'home_team_over_cover', 'home_team_under_cover', 'away_team_wins',
    'away_team_losses', 'away_team_tie', 'away_team_division_place', 'away_team_points_for',
    'away_team_points_against', 'away_team_cover', 'away_team_fail_cover', 'away_team_over_cover',
    'away_team_under_cover', 'home_team_completions', 'home_team_attempts', 'home_team_passing_yards',
    'home_team_passing_tds', 'home_team_interceptions', 'home_team_sacks', 'home_team_sack_yards',
    'home_team_sack_fumbles', 'home_team_sack_fumbles_lost', 'home_team_passing_air_yards',
    'home_team_passing_yards_after_catch', 'home_team_passing_first_downs', 'home_team_passing_epa',
    'home_team_passing_two_point_conversions', 'home_team_carries', 'home_team_rushing_yards', 'home_team_rushing_tds',
    'home_team_rushing_fumbles', 'home_team_rushing_fumbles_lost', 'home_team_rushing_first_downs',
    'home_team_rushing_epa', 'home_team_rushing_two_point_conversions', 'home_team_receiving_fumbles',
    'home_team_receiving_fumbles_lost', 'home_team_special_teams_tds', 'home_team_fantasy_points',
    'home_team_fantasy_points_ppr', 'away_team_completions', 'away_team_attempts', 'away_team_passing_yards',
    'away_team_passing_tds', 'away_team_interceptions', 'away_team_sacks', 'away_team_sack_yards',
    'away_team_sack_fumbles', 'away_team_sack_fumbles_lost', 'away_team_passing_air_yards',
    'away_team_passing_yards_after_catch', 'away_team_passing_first_downs', 'away_team_passing_epa',
    'away_team_passing_two_point_conversions', 'away_team_carries', 'away_team_rushing_yards', 'away_team_rushing_tds',
    'away_team_rushing_fumbles', 'away_team_rushing_fumbles_lost', 'away_team_rushing_first_downs',
    'away_team_rushing_epa', 'away_team_rushing_two_point_conversions', 'away_team_receiving_fumbles',
    'away_team_receiving_fumbles_lost', 'away_team_special_teams_tds', 'away_team_fantasy_points',
    'away_team_fantasy_points_ppr'
]

group_cols = ["recent_team", "season_type", "week"]

keep_cols = [
    "team_name", "playoff", "week", "completions", "attempts", "passing_yards", "passing_tds",
    "interceptions", "sacks", "sack_yards", "sack_fumbles", "sack_fumbles_lost", "passing_air_yards",
    "passing_yards_after_catch", "passing_first_downs", "passing_epa", "passing_2pt_conversions",
    "carries", "rushing_yards", "rushing_tds", "rushing_fumbles", "rushing_fumbles_lost",
    "rushing_first_downs", "rushing_epa", "rushing_2pt_conversions", "receiving_fumbles",
    "receiving_fumbles_lost", "special_teams_tds", "fantasy_points", "fantasy_points_ppr"
]


class NFLStatsAPI:
    """Class to extract nfl stats by week"""
    def __init__(self):
        load_dotenv()

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
            print(f"Finding standing data for team: {team_name}")
            response = requests.get(f"https://api.sportsdata.io/v3/nfl/scores/json/Standings/2023?key={self.api_key}")
            standings = response.json()

            # Extract correct team data
            for team in standings:
                if team["Team"] == team_abbreviations[team_name] or team["Name"] == team_name:
                    wins = team["Wins"]
                    losses = team["Losses"]
                    ties = team["Ties"]
                    place = team["DivisionRank"]
                    points_for = team["PointsFor"]
                    points_against = team["PointsAgainst"]
                    return [wins, losses, ties, place, points_for, points_against]
            return [0, 0, 0, 0, 0, 0]

        except Exception as e:
            print(f"Standings API call failed; Details: {e}")
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
            spread_wins = int(spread_record[0]) + int(spread_record[2])
            spread_losses = int(spread_record[1])

        except Exception as e:
            print(f"Betting Record API call failed; Details: {e}")
            return [0, 0, 0, 0]

        # Find over data
        try:
            print(f"Finding over point total record for team: {team_name}")
            point_total_df = pd.read_html("https://betiq.teamrankings.com/nfl/betting-trends/over-under-records/")[0]
            team_points_total_df = point_total_df[point_total_df["Team"] == team_markets[team_name]]
            team_points_total_df.reset_index(inplace=True, drop=True)
            over_record = team_points_total_df["Over Record"][0].split("-")
            over_wins = int(over_record[0]) + int(over_record[2])
            over_losses = int(over_record[1])

        except Exception as e:
            print(f"Points Total Record API call failed; Details: {e}")
            return [0, 0, 0, 0]

        return [spread_wins, spread_losses, over_wins, over_losses]

    def compile_team_stats(self, team_name, week, playoff):
        """Function to compile home team standings, betting, and statistical data"""
        standing_data = self.find_standing_data(team_name)
        betting_data = self.find_betting_data(team_name)
        statistical_data = self.extract_season_stats(team_abbreviations[team_name], week, playoff)

        compiled_data = standing_data + betting_data + statistical_data
        return compiled_data

    def collect_data_for_predictions(self, week, home_team, away_team, favorite, given_spread,
                                     given_total, stadium, playoff, neutral_site):
        """Function to collect data to feed models"""

        # Format provided general data about the game
        home_team_id = teams[home_team]
        away_team_id = teams[away_team]

        home_team_division_id = division_ids[divisions[home_team]]
        away_team_division_id = division_ids[divisions[away_team]]

        favorite_id = teams[favorite]

        stadium_id = stadiums[stadium]['id']
        stadium_city = stadiums[stadium]["city"]
        stadium_open_date = stadiums[stadium]["open_date"]
        stadium_roof_type = stadiums[stadium]["roof_type"]
        stadium_weather_type = stadiums[stadium]["weather_type"]
        stadium_capacity = stadiums[stadium]["capacity"]
        stadium_surface = stadiums[stadium]["surface"]
        stadium_latitude = stadiums[stadium]["latitude"]
        stadium_longitude = stadiums[stadium]["longitude"]

        general_data = [self.season, week, playoff, home_team_id, home_team_division_id, away_team_id,
                        away_team_division_id, favorite_id, given_spread, given_total, stadium_id, stadium_city,
                        stadium_open_date, stadium_roof_type, stadium_weather_type, stadium_capacity, stadium_surface,
                        stadium_latitude, stadium_longitude, neutral_site]

        # Collect home/away team season statistics
        home_team_stats = self.compile_team_stats(home_team, week, playoff)
        away_team_stats = self.compile_team_stats(away_team, week, playoff)

        # Combine and return data as data frame
        data = general_data + home_team_stats + away_team_stats
        df = pd.DataFrame(data=[data], columns=columns)
        return df
