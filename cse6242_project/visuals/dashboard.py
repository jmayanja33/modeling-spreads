import os

import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from plotly.graph_objs._figure import Figure
from dash import Dash, html, dcc, callback, Output, Input

from cse6242_project import PROJECT_ROOT
from cse6242_project.utilities.infer import get_score_graph_data
from cse6242_project.utilities import get_team_fullname


DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SCHEDULE_DATA = pd.read_csv(os.path.join(DATA_DIR, "2023_schedule.csv"))
EXPANDED_DATA = pd.read_csv(os.path.join(DATA_DIR, "expanded_data.csv"))


def create_fig(label: str,
               team1_score: int,
               team2_score: float) -> Figure:
    """Example function to createa a gauge plot figure.

    Args:
        team1 (str): Name of first team
        team2 (str): Name of second team
        mock_win_prob (float): Probability that first team wins.

    Returns:
        Figure with gauge chart for team win probability.
    """
    ratio = max(team1_score, team2_score) / (team1_score + team2_score)
    if team1_score > team2_score:
        ratio = 1 - ratio
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=ratio,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": label, "font": {"size": 18}},
            gauge={
                "axis": {"range": [None, 1], "tickwidth": 1, "tickcolor": "black"},
                "bar": {"color": "#a5d5d8" if team1_score < team2_score else "#f4777f"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 0.5], "color": "#cf3759"},
                    {"range": [0.5, 1], "color": "#4771b2"},
                ],
            },
        ),
    )
    return fig


# read in required data set
df = pd.read_csv(os.path.join(DATA_DIR, "expanded_data.csv"))

# Get array of current team
current_teams = [*map(get_team_fullname, SCHEDULE_DATA.Teams.unique())]

# create dropdown objects to select each team.
team1_dropdown = html.Div(
    [
        html.Label(
            "Select Your Team",
            htmlFor="team1-dropdown",
            style={"margin": "auto", "font-weight": "bold", "textAlign": "center"},
        ),
        dcc.Dropdown(
            current_teams,
            current_teams[0],
            id="team1-dropdown",
            style={"margin": "auto", "textAlign": "center"},
        ),
    ],
    style={"margin": "auto", "width": "200%"},
)

# initialize app. Include bootstrap libraries.
dashapp = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

gauge_size = 600
# define the app layout
dashapp.layout = html.Div(
    [
        html.Div(
            children=[
                html.H1(
                    children="Beat the Bookie",
                    style={"textAlign": "center"},
                ),
                dbc.Row(
                    [dbc.Col(team1_dropdown)],
                    style={
                        "display": "flex",
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
                    ],
                    style={
                        "margin": "auto",
                        "width": "25%",
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
def update_graph(team1: str):
    model_output = get_score_graph_data(team1)
    if model_output is not None:
        team1, team1_score, team2, team2_score = model_output
        label = f"{team1}: {int(team1_score)} | {team2}: {int(team2_score)}"
    else:
        label, team1_score, team2_score = (f'{team1} is on a BYE week', 1, 1)
    fig1 = create_fig(label, int(team1_score), int(team2_score))
    return fig1


def main():
    dashapp.run(debug=True)
