import os
from typing import Tuple
import numpy as np

from sklearn.pipeline import Pipeline
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from plotly.graph_objs._figure import Figure
from dash import Dash, html, dcc, callback, Output, Input

from cse6242_project import PROJECT_ROOT
from cse6242_project.utilities.infer import get_schedule_data, get_tree_predictions
from cse6242_project.utilities import get_team_fullname, get_team_abbrv
from cse6242_project.utilities.data import load_weekly_stats
from cse6242_project.utilities import load_model


DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SCHEDULE_DATA = pd.read_csv(os.path.join(DATA_DIR, "2023_schedule.csv"))
EXPANDED_DATA = pd.read_csv(os.path.join(DATA_DIR, "expanded_data.csv"))


def on_load() -> Tuple[pd.DataFrame, Pipeline]:
    try:
        # Load github data
        weekly_stats_df = load_weekly_stats()
        # Add schedule data.
        weekly_stats_df['is_home'], weekly_stats_df['opponent'], __ = zip(*weekly_stats_df.team.apply(get_schedule_data))
        weekly_stats_df.fillna('BYE', inplace=True)
        weekly_stats_df.set_index('team', inplace=True)
    except:
        # Use stored backup data for demo if github cannot be reached
        weekly_stats_df = pd.read_csv(os.path.join(DATA_DIR, "demo_backup_weekly_stats.csv"), index_col='team')
        weekly_stats_df['is_home'] = weekly_stats_df.is_home.apply(lambda x: {'True': True, 'False': False}.get(x, x))
        UserWarning("Github cannot be reached for pulling current data. Using demo dataset.")

    # load model
    model = load_model('rf_regressor_pipeline3.pkl')

    return weekly_stats_df, model


WEEKLY_STATS, MODEL = on_load()


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

def create_overlap_hist(team1_name, data1, team2_name, data2):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data1, name=team1_name))
    fig.add_trace(go.Histogram(x=data2, name=team2_name))

    # TODO Update figure layout
    # TODO x-label = predicted score
    # TODO y-label = count
    fig.update_layout(barmode='overlay',
                      title='Team Score Predictions'
                      )
    
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)

    return fig

def create_spread_hist(data):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data))

    # TODO Update figure layout
    # TODO x-label = predicted score
    # TODO y-label = count
    fig.update_layout(title='Predicted Spread')

    return fig

def create_single_hist(data):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data))

    # TODO Update figure layout
    # TODO x-label = predicted score
    # TODO y-label = count
    fig.update_layout(title="Predicted Total Score")

    return fig

def create_feature_barchart(features, team1, data1, team2, data2):
    # TODO: x-label: feature
    # TODO: y-label: relative performance
    # TODO: title: Decisive features
    fig = go.Figure(data=[
        go.Bar(name=team1, x=features, y=data1),
        go.Bar(name=team2, x=features, y=data2),
    ])
    fig.update_layout(barmode='group')
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

plot_width = 800
plot_height = 400
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
                                "height": plot_height,
                                "width": plot_width,
                                "scale": "100%"
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
def get_team_stats(team_abbrv: str) -> pd.Series:
    return WEEKLY_STATS.loc[team_abbrv]


@callback(Output("graph-content1", "figure"), Input("team1-dropdown", "value"))
def update_graph(team1: str):
    # Get team 1 stats
    team1_abbrv = get_team_abbrv(team1)
    team1_stats = get_team_stats(team1_abbrv)

    # Get team 2 name
    team2_abbrv = team1_stats.opponent

    if team2_abbrv == "BYE":
        # Bye week logic
        return create_fig(f'{team1} has a BYE week', 0, 1)

    # Get team 2 stats
    team2_stats = get_team_stats(team2_abbrv)

    # Build team1 input vector: team1_is_home, team1_stats, team2_stats
    team1_input_vector = pd.concat(
        [
            team1_stats[['is_home']],
            team1_stats.drop(['opponent', 'is_home']).add_suffix('_mean'),
            team2_stats.drop(['opponent', 'is_home']).add_suffix('_mean_other')
        ]
    ).to_frame().T

    # Build team2 input vector: team2_is_home, team2_stats, team1_stats
    team2_input_vector = pd.concat(
        [
            team2_stats[['is_home']],
            team2_stats.drop(['opponent', 'is_home']).add_suffix('_mean'),
            team1_stats.drop(['opponent', 'is_home']).add_suffix('_mean_other')
        ]
    ).to_frame().T

    # Get prediction and histogram data for team 1
    team1_pred_score = MODEL.predict(team1_input_vector)[0]
    team1_tree_preds = get_tree_predictions(MODEL, team1_input_vector)

    # Get prediction and histogram data for team 2
    team2_pred_score = MODEL.predict(team2_input_vector)[0]
    team2_tree_preds = get_tree_predictions(MODEL, team2_input_vector)

    # Compute total points and spread
    total_points = team1_pred_score + team2_pred_score
    spread = team1_pred_score - team2_pred_score

    # Get df.describe on histogram data
    tree_df = pd.DataFrame({'team1_score': team1_tree_preds, 'team2_score': team2_tree_preds})
    tree_df['spread'] = tree_df['team1_score'] - tree_df['team2_score']
    tree_df['total_score'] = tree_df['team1_score'] + tree_df['team2_score']
    tree_stats_df = tree_df.describe()

    # Get important features for each team to build vertical barchart
    feature_idx = np.argsort(MODEL.named_steps['rf'].feature_importances_)[::-1][:5]
    important_feature_names = team1_input_vector.iloc[:,feature_idx].columns.to_list()
    team1_important_feature_values = team1_input_vector.loc[:, important_feature_names].astype(float).squeeze()
    team2_important_feature_values = team2_input_vector.loc[:, important_feature_names].astype(float).squeeze()
    max_vals = np.array(
        [max(team1_important_feature_values[f], team2_important_feature_values[f]) for f in important_feature_names]
    )
    team1_important_feature_values /= max_vals
    team2_important_feature_values /= max_vals
    
    # team1_important_feature_values['team'] = team1_abbrv
    # team2_important_feature_values['team'] = team2_abbrv
    # barchar_df = pd.concat([team1_important_feature_values, team2_important_feature_values])

    ### Gathering different widget outputs ###

    # Build output for total points
    #TODO

    # Build output for spread
    #TODO
    
    # Build histogram figures
    #TODO

    # Build output for histogram describes
    #TODO

    # Build output for feature importance barchart
    #TODO

    fig1 = create_overlap_hist(team1_abbrv, team1_tree_preds, team2_abbrv, team2_tree_preds)
    fig1 = create_single_hist(tree_df['spread'])
    fig1 = create_single_hist(tree_df['total_score'])
    print(team1_important_feature_values.to_list())
    fig1 = create_feature_barchart(important_feature_names, team1_abbrv, team1_important_feature_values.to_list(), team2_abbrv, team2_important_feature_values.to_list())
    # fig1 = create_fig(f'{team1_abbrv} Score: {team1_pred_score} | {team2_abbrv} Score: {team2_pred_score}', int(team1_pred_score), int(team2_pred_score))
    return fig1


def main():
    dashapp.run(debug=True)

if __name__ == "__main__":
    main()
