import os
from typing import List, Tuple
from random import random

import pandas as pd
from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs._figure import Figure


PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def mock_model(
    team1: str, team2: str, teams: List[str]
) -> Tuple[str, float, str, float]:
    if team1 not in teams:
        raise Exception(f"{team1} not a team.")
    if team2 not in teams:
        raise Exception(f"{team1} not a team.")
    mock_win_prob = random()
    mock_lose_prob = 1 - mock_win_prob
    return team1, mock_win_prob, team2, mock_lose_prob


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
            title={"text": f"Probability {team1} Beats {team2}", "font": {"size": 14}},
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
df = pd.read_csv(os.path.join(PROJECT_PATH, "Data", "expanded_data.csv"))

# Get array of current team
current_teams = df[df.year == df.year.max()].home_team_name.unique()

# create dropdown objects to select each team.
team1_dropdown = html.Div(
    [
        html.Label(
            "Select Team 1",
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

team2_dropdown = html.Div(
    [
        html.Label(
            "Select Team 2",
            htmlFor="team2-dropdown",
            style={"font-weight": "bold", "text-align": "center"},
        ),
        dcc.Dropdown(
            current_teams,
            current_teams[1],
            id="team2-dropdown",
            style={"margin": "auto", "textAlign": "center"},
        ),
    ],
    style={"margin": "auto", "width": "50%"},
)

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
                    [dbc.Col(team1_dropdown), dbc.Col(team2_dropdown)],
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


@callback(
    [
        Output("graph-content1", "figure"),
        Output("graph-content2", "figure"),
        Output("graph-content3", "figure"),
        Output("graph-content4", "figure"),
    ],
    [Input("team1-dropdown", "value"), Input("team2-dropdown", "value")],
)
def update_graph(team1: str, team2: str) -> Tuple[Figure, Figure, Figure, Figure]:
    team1, mock_win_prob, team2, __ = mock_model(team1, team2, current_teams)
    fig1 = create_fig(team1, team2, mock_win_prob)

    team1, mock_win_prob, team2, __ = mock_model(team1, team2, current_teams)
    fig2 = create_fig(team1, team2, mock_win_prob)

    team1, mock_win_prob, team2, __ = mock_model(team1, team2, current_teams)
    fig3 = create_fig(team1, team2, mock_win_prob)

    team1, mock_win_prob, team2, __ = mock_model(team1, team2, current_teams)
    fig4 = create_fig(team1, team2, mock_win_prob)

    return fig1, fig2, fig3, fig4


app.run(debug=True)
