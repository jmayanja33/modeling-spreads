import os
from typing import List, Tuple, Dict, Any
from argparse import ArgumentParser
from random import random

import requests
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from datetime import datetime
from plotly.graph_objs._figure import Figure
from dash import Dash, html, dcc, callback, Output, Input


API_HOST = "127.0.0.1"
API_PORT = 5000
API_VERSION = "v1"
API_URL = f"http://{API_HOST}:{API_PORT}/api/{API_VERSION}"
PREDICT_URL = API_URL + "/predict"

HERE = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(HERE, "..", "Data")
SCHEDULE_DATA = pd.read_csv(os.path.join(DATA_DIR, "2023_schedule.csv"))
EXPANDED_DATA = pd.read_csv(os.path.join(DATA_DIR, "expanded_data.csv"))
TODAY = datetime.now().date()


def get_week():
    col_idx = {
        datetime.strptime(c, "%m/%d/%y").date(): i
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
        "week": week == week if week != "playoffs" else 18,
        "opponent": opponent,
        "is_home": is_home,
        "stadium": stadium,
        "playoff": week == "playoff",
        "favorite": favorite,
        "given_spread": given_spread,
        "given_total": given_total,
    }


def infer(team1):
    weekly_info = get_weekly_info(team1)
    team2 = weekly_info["opponent"]
    favorite = weekly_info["favorite"]
    given_spread = weekly_info["given_spread"]
    given_total = weekly_info["given_total"]
    stadium = weekly_info["stadium"]
    playoff = weekly_info["playoff"]
    week = weekly_info["week"]
    neutral_site = False
    query_url = (
        PREDICT_URL
        + f"/{week}/{team1}/{team2}/{favorite}/{given_spread}/{given_total}/{stadium}/{playoff}/{neutral_site}"
    )
    try:
        r = requests.get(query_url)
        if str(r.status_code).startswith("2"):
            return r.json()
    except Exception as e:
        print(e)

    return team2, random()


def get_graph_data(team_name):
    return infer(team_name)


def create_fig(team1: str, team2: str, mock_win_prob: float) -> Figure:
    """Example function to createa a gauge plot figure.

    Args:
        team1 (str): Name of first team
        team2 (str): Name of second team
        mock_win_prob (float): Probability that first team wins.

    Returns:
        Figure with gauge chart for team win probability.
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=mock_win_prob,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"{team1}   |   {team2}", "font": {"size": 14}},
            gauge={
                "axis": {"range": [None, 1], "tickwidth": 1, "tickcolor": "black"},
                "bar": {"color": "#a5d5d8" if mock_win_prob > 0.5 else "#f4777f"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 0.5], "color": "#cf3759"},
                    {"range": [0.5, 1], "color": "#4771b2"},
                ],
            },
        ),
        # go.Layout(height=400, width=400),
    )
    return fig


# read in required data set
df = pd.read_csv(os.path.join(DATA_DIR, "expanded_data.csv"))

# Get array of current team
current_teams = SCHEDULE_DATA.Teams.unique()

# create dropdown objects to select each team.
team1_dropdown = html.Div(
    [
        html.Label(
            "Select Your Team",
            htmlFor="team1-dropdown",
            style={"font-weight": "bold", "text-align": "center"},
        ),
        dcc.Dropdown(
            current_teams,
            current_teams[0],
            id="team1-dropdown",
            style={"margin": "auto", "textAlign": "center"},
        ),
    ],
    style={"margin": "auto", "width": "50%"},
)

# team2_dropdown = html.Div(
#     [
#         html.Label(
#             "Select Team 2",
#             htmlFor="team2-dropdown",
#             style={"font-weight": "bold", "text-align": "center"},
#         ),
#         dcc.Dropdown(
#             current_teams,
#             current_teams[1],
#             id="team2-dropdown",
#             style={"margin": "auto", "textAlign": "center"},
#         ),
#     ],
#     style={"margin": "auto", "width": "50%"},
# )

# initialize app. Include bootstrap libraries.
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

gauge_size = 400
# define the app layout
app.layout = html.Div(
    [
        html.Div(
            children=[
                html.H1(
                    children="Beat the Bookie",
                    style={"textAlign": "center"},
                ),
                dbc.Row(
                    [dbc.Col(team1_dropdown)],  # dbc.Col(team2_dropdown)],
                    style={
                        "display": "flex",
                        "align-items": "center",
                        "justify-content": "center",
                    },
                ),
            ]
        ),
        html.Div(
            children=[
                dbc.Row(
                    [
                        dcc.Graph(
                            id="graph-content1",
                            style={
                                "height": gauge_size,
                                "width": gauge_size,
                            },
                        ),
                        dcc.Graph(
                            id="graph-content2",
                            style={
                                "height": gauge_size,
                                "width": gauge_size,
                            },
                        ),
                    ],
                    style={
                        "margin": "auto",
                        "width": "50%",
                    },
                ),
                dbc.Row(
                    [
                        dcc.Graph(
                            id="graph-content3",
                            style={
                                "height": gauge_size,
                                "width": gauge_size,
                            },
                        ),
                        dcc.Graph(
                            id="graph-content4",
                            style={
                                "height": gauge_size,
                                "width": gauge_size,
                            },
                        ),
                    ],
                    style={
                        "margin": "auto",
                        "width": "50%",
                        "margin-top": "1px",
                        "margin-bottom": "1px",
                    },
                ),
            ],
        ),
    ]
)


"""
To add a new graph to the dashboard:
    Create a new function to create the figure you need.
    Add an Output object with the graph id to the output list.
    Create a Figure object using the new function you created.
    Return the Figure in the appropriate order,
        relative to the output list in callback decorator.
"""


@callback(Output("graph-content1", "figure"), Input("team1-dropdown", "value"))
def update_graph(team1: str):  # , team2: str) -> Tuple[Figure, Figure]:
    # team1, mock_win_prob, team2, __ = infer(team1, team2)
    team2, model_output = infer(team1)
    print(team1, team2, model_output)
    fig1 = create_fig(team1, team2, model_output)

    # team1, mock_win_prob, team2, __ = infer(team1, team2)
    # fig2 = create_fig(team1, team2, mock_win_prob)

    return fig1


def main():
    parser = ArgumentParser()

    app.run(debug=True)


if __name__ == "__main__":
    main()
