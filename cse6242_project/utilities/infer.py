import os
from typing import Tuple, Dict, Any

import pandas as pd
from datetime import datetime
from cse6242_project import PROJECT_ROOT
from cse6242_project.utilities import load_model
from cse6242_project.utilities import WeeklyRanking


rf_score_model = load_model('rf_regressor_score.pkl')
weekly_rankings = WeeklyRanking()


DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SCHEDULE_DATA = pd.read_csv(os.path.join(DATA_DIR, "2023_schedule.csv"))
EXPANDED_DATA = pd.read_csv(os.path.join(DATA_DIR, "expanded_data.csv"))
TEAM_NAME_DATA = pd.read_csv(os.path.join(DATA_DIR, "nfl_teams.csv"))
TODAY = datetime.now().date()

team_name_map = {k: v for k, v in zip(TEAM_NAME_DATA.team_name_short, TEAM_NAME_DATA.team_name)}


def get_predicted_score(team_name):
    weekly_ranking_data = weekly_rankings.get_rankings(
        team_name
    )
    return rf_score_model.predict(weekly_ranking_data)[0]



def get_week():
    col_idx = {
        datetime.strptime(c, "%m/%d/%Y").date(): i
        for i, c in enumerate(SCHEDULE_DATA.columns.tolist()[1:])
    }
    today = datetime.now().date()
    for idx_date in col_idx:
        if today < idx_date:
            break
    if today > idx_date:
        week = "playoffs"
    else:
        week = col_idx[idx_date] + 1
    return week


def get_schedule_data(team_name: str):
    week = get_week()
    team1_schedule = SCHEDULE_DATA[SCHEDULE_DATA.Teams == team_name]
    if week != "playoffs":
        opponent = team1_schedule.iloc[:, week].squeeze()
    else:
        opponent = None
    if (opponent is not None) and (opponent != "BYE"):
        home_indicator, opponentname = opponent.split()
        is_home = home_indicator == "vs."
    else:
        is_home = None
        opponentname = None
    return is_home, opponentname, week


def get_stadium(team_name: str):
    return (
        EXPANDED_DATA[
            (EXPANDED_DATA.year == EXPANDED_DATA.year.max())
            & (EXPANDED_DATA.home_team_name == team_name)
        ]
        .stadium_name.iloc[:1]
        .squeeze()
    )


def get_betting_odds(team_name):
    return True, 10, 21


def get_weekly_info(team_name: str) -> Dict[str, Any]:
    is_home, opponent, week = get_schedule_data(team_name)
    stadium = get_stadium(team_name if is_home else opponent)
    favorite, given_spread, given_total = get_betting_odds(team_name)
    return {
        "week": week if week != "playoffs" else 18,
        "opponent": opponent,
        "is_home": is_home,
        "stadium": stadium,
        "playoff": week == "playoff",
        "favorite": favorite,
        "given_spread": given_spread,
        "given_total": given_total,
    }


def get_score_graph_data(team1: str) -> Tuple[str, int, str, int]:
    weekly_info = get_weekly_info(team1)
    team2 = team_name_map[weekly_info["opponent"]]
    try:
        team1_response = get_predicted_score(team1)
    except Exception as e:
        print(f'Could not determine team {team1} score: {e}')
        team1_response = 0
    try:
        team2_response = get_predicted_score(team2)
    except Exception as e:
        print(f'Could not determine team {team2} score: {e}')
        team2_response = 0
    team1_score = team1_response
    team2_score = team2_response
    return team1, team1_score, team2, team2_score