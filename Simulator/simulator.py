"""
Script to simulate every game from all seasons. Extra stats will be extracted from each game and saved as features
"""

import pandas as pd
from root_path import ROOT_PATH
from divisions import *
from stadiums import stadiums


def check_for_playoff(value):
    """Function to check if a game is a playoff game"""
    if value == False:
        return 0
    return 1


def check_for_neutral_site(value):
    """Function to check if a game is being played at a neutral site"""
    if value == False:
        return 0
    return 1


def calculate_actual_spread(given_spread, home_team, home_score, away_score, favorite):
    """
    Function to calculate the actual spread of the game and determine if the spread hit
    :param given_spread: Spread set by sports book
    :param home_team: ID of home team
    :param home_score: Number of points scored in game by home team
    :param away_score: Number of points scored in game by away team
    :param favorite: ID of the team that is the favorite
    :return: `spread`: Spread of the final score; `covered`: 1 or 0 to indicate if the favorite covered
    """

    # Calculate spread
    if favorite == home_team:
        spread = away_score - home_score
    else:
        spread = home_score - away_score

    # Evaluate spread, return if favorite covered w/actual spread
    if spread <= given_spread:
        return spread, 1
    return spread, 0


def evaluate_point_total(given_total, home_score, away_score):
    """
    Function to determine if game went over the predicted point total
    :param given_total: Number of total points set by sports book
    :param home_score: Number of points scored in game by home team
    :param away_score: Number of points scored in game by away team
    :return: 1 or 0 to indicate if the over hit
    """
    point_total = home_score + away_score
    if point_total >= given_total:
        return 1
    else:
        return 0


class Simulator:
    """Class to simulate all nfl seasons in the data set"""

    def __init__(self, start_season=2002, end_season=2023):
        self.data = pd.read_csv(f"{ROOT_PATH}/Data/spreadspoke_scores.csv")
        self.start_season = start_season
        self.end_season = end_season
        self.simulated_data = []
        self.division_standings = None

    def check_for_clinched_playoffs(self, team):
        """Function to check if a team has clinched a playoff spot"""
        raise NotImplementedError

    def update_standings_game(self, playoff, home_team_name, home_team_id, home_score, away_team_name,
                              away_score, favorite, given_spread, over_covered):
        """
        Function to update standings after each game
        :param playoff: 1 or 0 to indicate if the game is a playoff game
        :param home_team_name: Name of the home team
        :param home_team_id: ID of the home team
        :param home_score: Number of points scored in game by home team
        :param away_team_name: Name of the away team
        :param away_score: Number of points scored in game by away team
        :param favorite: Team ID of the favorite
        :param given_spread: Spread set by sports book
        :param over_covered: 1 or 0 to indicate if the over hit
        :return: None
        """
        if playoff == 0:
            home_team_division = divisions[home_team_name]
            away_team_division = divisions[away_team_name]
            home_team_record = self.division_standings[home_team_division][home_team_name]
            away_team_record = self.division_standings[away_team_division][away_team_name]

            # Home team wins
            if home_score > away_score:
                home_team_record["win"] += 1
                away_team_record["loss"] += 1
                if home_team_division == away_team_division:
                    home_team_record["division win"] += 1
                    away_team_record["division loss"] += 1
                if away_team_name not in home_team_record["teams beaten"].keys():
                    home_team_record["teams beaten"][away_team_name] = 0
                home_team_record["teams beaten"][away_team_name] += 1

            # Away team wins
            elif home_score < away_score:
                home_team_record["loss"] += 1
                away_team_record["win"] += 1
                if home_team_division == away_team_division:
                    away_team_record["division win"] += 1
                    home_team_record["division loss"] += 1
                if home_team_name not in away_team_record["teams beaten"].keys():
                    away_team_record["teams beaten"][home_team_name] = 0
                away_team_record["teams beaten"][home_team_name] += 1

            # Tie
            else:
                home_team_record["tie"] += 1
                away_team_record["tie"] += 1
                if home_team_division == away_team_division:
                    home_team_record["division tie"] += 1
                    away_team_record["division tie"] += 1

            # Determine cover
            if favorite == home_team_id:
                if away_score - home_score > given_spread:
                    home_team_record["cover"] += 1
                elif away_score - home_score < given_spread:
                    away_team_record["cover"] += 1
                else:
                    home_team_record["cover"] += 1
                    away_team_record["cover"] += 1
            else:
                if home_score - away_score > given_spread:
                    away_team_record["cover"] += 1
                elif home_score - away_score < given_spread:
                    home_team_record["cover"] += 1
                else:
                    home_team_record["cover"] += 1
                    away_team_record["cover"] += 1

            # Determine over cover
            if over_covered == 1:
                home_team_record["over cover"] += 1
                away_team_record["over cover"] += 1

            # Track points
            home_team_record["points for"] += home_score
            away_team_record["points for"] += away_score
            home_team_record["points against"] += away_score
            away_team_record["points against"] += home_score

    def update_standings_week(self, playoff):
        """
        Function to update standings after each week
        :param playoff: 1 or 0 to indicate if the game was a playoff game
        :return: None
        """
        if playoff == 0:
            for division in self.division_standings.keys():
                # Create a list of tuples with each team's record
                standing_list = []
                for team in self.division_standings[division].keys():
                    team_stats = self.division_standings[division][team]
                    try:
                        team_division_win_percent = team_stats["division win"] / (team_stats["division win"] + team_stats["division loss"] + team_stats["division tie"])
                    except ZeroDivisionError:
                        team_division_win_percent = 0
                    team_record = (
                        team, team_stats["win"], team_stats["loss"], team_stats["tie"], team_stats["points for"],
                        team_stats["points against"],team_division_win_percent
                    )
                    standing_list.append(team_record)

                # Sort list
                sorted_standings = sorted(standing_list, key=lambda i: (i[1], i[3], i[2], i[6], i[4], i[5]),
                                          reverse=True)

                # Final round of sorting
                for i in range(len(sorted_standings)-1):
                    team_1 = sorted_standings[i]
                    team_2 = sorted_standings[i+1]
                    team_1_opponent_wins = self.division_standings[division][team_1[0]]["teams beaten"]
                    team_2_opponent_wins = self.division_standings[division][team_2[0]]["teams beaten"]

                    # Tiebreaker: elevate team with a better head to head record
                    if team_1[1:3] == team_2[1:3]:
                        if team_1 in team_2_opponent_wins.keys() and team_2 in team_1_opponent_wins.keys():
                            if team_2_opponent_wins[team_1] > team_1_opponent_wins[team_2]:
                                sorted_standings[i] = team_2
                                sorted_standings[i+1] = team_1
                        elif team_2 in team_1_opponent_wins.keys():
                            sorted_standings[i] = team_2
                            sorted_standings[i + 1] = team_1
                        else:
                            continue

                    #  Tiebreaker: Raise teams with equal wins that have played fewer games
                    if team_1[1] == team_2[1] and sum(team_1[1:3]) > sum(team_2[1:3]):
                        sorted_standings[i] = team_2
                        sorted_standings[i + 1] = team_1

                # Update final standings
                print(f"Division: {division}")
                for i in range(len(sorted_standings)):
                    team = sorted_standings[i][0]
                    wins = sorted_standings[i][1]
                    losses = sorted_standings[i][2]
                    ties = sorted_standings[i][3]
                    self.division_standings[division][team]["place"] = i+1
                    print(f"{i+1}.) {team} ({wins}-{losses}-{ties})")
                print("\n")

    def simulate_week(self, season_data, week):
        """
        Function to simulate a week
        :param season_data: Dataframe containg data from all games in a single NFL season
        :param week: Week of the season to be simulated
        :return: None
        """
        year = season_data["schedule_season"][0]

        # Filter data to only include data form week
        week_data = season_data[season_data["schedule_week"] == week]
        week_data.reset_index(inplace=True, drop=True)

        # Determine if playoffs
        playoff_week = check_for_playoff(week_data["schedule_playoff"][0])

        # Iterate through games to simulate week
        for i in range(len(week_data)):
            playoff_game = check_for_playoff(week_data["schedule_playoff"][i])
            # Home team data
            home_team_name = week_data["team_home"][i]
            home_team_division_name = divisions[home_team_name]
            home_team_id = teams[home_team_name]
            home_team_division_id = division_ids[home_team_division_name]
            home_team_wins = self.division_standings[home_team_division_name][home_team_name]["win"]
            home_team_losses = self.division_standings[home_team_division_name][home_team_name]["loss"]
            home_team_tie = self.division_standings[home_team_division_name][home_team_name]["tie"]
            home_team_cover = self.division_standings[home_team_division_name][home_team_name]["cover"]
            home_team_over_cover = self.division_standings[home_team_division_name][home_team_name]["over cover"]
            home_team_division_place = self.division_standings[home_team_division_name][home_team_name]["place"]
            home_team_points_for = self.division_standings[home_team_division_name][home_team_name]["points for"]
            home_team_points_against = self.division_standings[home_team_division_name][home_team_name]["points against"]
            # home_team_clinched_playoffs = self.check_for_clinched_playoffs(home_team_name)
            home_score = int(week_data["score_home"][i])

            # Away team data
            away_team_name = week_data["team_away"][i]
            away_team_division_name = divisions[away_team_name]
            away_team_division_id = division_ids[away_team_division_name]
            away_team_id = teams[away_team_name]
            away_team_wins = self.division_standings[away_team_division_name][away_team_name]["win"]
            away_team_losses = self.division_standings[away_team_division_name][away_team_name]["loss"]
            away_team_tie = self.division_standings[away_team_division_name][away_team_name]["tie"]
            away_team_cover = self.division_standings[away_team_division_name][away_team_name]["cover"]
            away_team_over_cover = self.division_standings[away_team_division_name][away_team_name]["over cover"]
            away_team_division_place = self.division_standings[away_team_division_name][away_team_name]["place"]
            away_team_points_for = self.division_standings[away_team_division_name][away_team_name]["points for"]
            away_team_points_against = self.division_standings[away_team_division_name][away_team_name]["points against"]
            # away_team_clinched_playoffs = self.check_for_clinched_playoffs(away_team_name)
            away_score = int(week_data["score_away"][i])

            # Betting data
            favorite = teams[week_data["team_favorite_id"][i]]
            given_spread = float(week_data["spread_favorite"][i])
            given_total = float(week_data["over_under_line"][i])
            actual_spread, favorite_covered = calculate_actual_spread(given_spread, home_team_id, home_score, away_score,
                                                                      favorite)
            over_covered = evaluate_point_total(given_total, home_score, away_score)
            total_points = home_score + away_score

            # Stadium data
            stadium_name = week_data["stadium"][i]
            stadium_id = stadiums[stadium_name]["id"]
            stadium_city = stadiums[stadium_name]["city"]
            stadium_open_date = stadiums[stadium_name]["open_date"]
            stadium_roof_type = stadiums[stadium_name]["roof_type"]
            stadium_weather_type = stadiums[stadium_name]["weather_type"]
            stadium_capacity = stadiums[stadium_name]["capacity"]
            stadium_surface = stadiums[stadium_name]["surface"]
            stadium_latitude = stadiums[stadium_name]["latitude"]
            stadium_longitude = stadiums[stadium_name]["longitude"]
            # stadium_azimuth_angle = stadiums[stadium_name]["azimuth_angle"]
            # stadium_elevation = stadiums[stadium_name]["elevation"]
            neutral_site = check_for_neutral_site(week_data["stadium_neutral"][i])
            if stadium_roof_type == 0:
                kickoff_temperature = 70
                kickoff_wind = 0
                kickoff_humidity = 0
            else:
                kickoff_temperature = week_data["weather_temperature"][i]
                kickoff_wind = week_data["weather_wind_mph"][i]
                kickoff_humidity = week_data["weather_humidity"][i]

            # Save game data
            game_data = (year, week, playoff_game, home_team_name, home_team_division_name, home_team_id,
                         home_team_division_id, home_team_wins, home_team_losses, home_team_tie, home_team_cover,
                         home_team_over_cover, home_team_division_place, home_score, home_team_points_for,
                         home_team_points_against, away_team_name, away_team_division_name, away_team_id,
                         away_team_division_id, away_team_wins, away_team_losses, away_team_tie, away_team_cover,
                         away_team_over_cover, away_team_division_place, away_score, away_team_points_for,
                         away_team_points_against, favorite, given_spread, given_total, stadium_name, stadium_id,
                         stadium_city, stadium_open_date, stadium_roof_type, stadium_weather_type, stadium_capacity,
                         stadium_surface, stadium_latitude, stadium_longitude,
                         neutral_site, kickoff_temperature, kickoff_wind, kickoff_humidity, actual_spread,
                         favorite_covered, over_covered, total_points)

            self.simulated_data.append(game_data)

            # Print game results
            if home_score > away_score:
                print(f"{year} Season, WEEK {week}: {home_team_name} def. {away_team_name}: {home_score}-{away_score}")
                if week == "Superbowl":
                    print(f"{home_team_name} won super bowl!\n")
            elif away_score > home_score:
                print(f"{year} Season, WEEK {week}: {away_team_name} def. {home_team_name}: {away_score}-{home_score}")
                if week == "Superbowl":
                    print(f"{away_team_name} won super bowl!\n")
            else:
                print(f"{year} Season, WEEK {week}: {away_team_name} tied {home_team_name}: {away_score}-{home_score}")

            # Update standings
            self.update_standings_game(playoff_game, home_team_name, home_team_id, home_score, away_team_name,
                                       away_score, favorite, given_spread, over_covered)

        # Update standings
        if playoff_week == 1:
            print(f"\nFinished week {week}\n")
        else:
            print(f"\nFinished week {week}; Standings:")
        self.update_standings_week(playoff_week)

    def simulate_season(self, year):
        """
        Function to simulate a single season
        :param year: Year of NFL season to be simulated
        :return: None
        """

        # Filter data to only include games from season
        season_data = self.data[self.data["schedule_season"] == year]
        season_data.reset_index(inplace=True, drop=True)

        # Initialize divisions
        self.division_standings = {
            "AFC East": {},
            "AFC West": {},
            "AFC North": {},
            "AFC South": {},
            "NFC East": {},
            "NFC West": {},
            "NFC North": {},
            "NFC South": {}
        }

        # Add teams to standings
        teams = set(season_data["team_home"].unique()).intersection(set(season_data["team_home"].unique()))
        for team in teams:
            team_division = divisions[team]
            self.division_standings[team_division][team] = {"win": 0, "loss": 0, "tie": 0, "cover": 0, "over cover": 0,
                                                            "division win": 0, "division loss": 0, "division tie": 0,
                                                            "points for": 0, "points against": 0, "place": 1,
                                                            "teams beaten": {}}

        # Iterate through each week to simulate season:
        if year < 2021:
            weeks = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                     'Wildcard', 'Division', 'Conference', 'Superbowl']
        else:
            weeks = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                     'Wildcard', 'Division', 'Conference', 'Superbowl']

        for week in weeks:
            print(f"Simulating data for {year} Season, Week {week}")
            self.simulate_week(season_data, week)

    def simulate_all_seasons(self):
        """Function to simulate all seasons defined"""

        # Simulate all seasons
        for i in range(self.start_season, self.end_season):
            print(f"Simulating games for season: {i}")
            self.simulate_season(i)

        columns = [
            "year", "week", "playoff_game", "home_team_name", "home_team_division_name", "home_team_id",
            "home_team_division_id", "home_team_wins", "home_team_losses", "home_team_tie", "home_team_cover",
            "home_team_over_cover", "home_team_division_place", "home_score", "home_team_points_for",
            "home_team_points_against", "away_team_name", "away_team_division_name", "away_team_id",
            "away_team_division_id", "away_team_wins", "away_team_losses", "away_team_tie", "away_team_cover",
            "away_team_over_cover", "away_team_division_place", "away_score", "away_team_points_for",
            "away_team_points_against", "favorite", "given_spread", "given_total", "stadium_name", "stadium_id",
            "stadium_city", "stadium_open_date", "stadium_roof_type", "stadium_weather_type", "stadium_capacity",
            "stadium_surface", "stadium_latitude", "stadium_longitude",
            "neutral_site", "kickoff_temperature", "kickoff_wind", "kickoff_humidity", "actual_spread",
            "favorite_covered", "over_covered", "total_points"
        ]

        # Save new data to csv file
        print("Saving data to file")
        df = pd.DataFrame(self.simulated_data, columns=columns)
        df.to_csv(f"{ROOT_PATH}/Data/expanded_data.csv", index=False)


# Main to run simulations
if __name__ == '__main__':
    simulator = Simulator()
    simulator.simulate_all_seasons()
