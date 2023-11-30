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

    fig.update_layout(
        barmode='overlay',
        title='Team Score Predictions',
        xaxis_title='Predicted Score',
        yaxis_title='Count',
        title_x=0.5,
        title_y=0.9,
        font_size=30,
    )

    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)

    return fig

def create_spread_hist(data):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data))
    fig.update_layout(
        title='Predicted Spread Distribution',
        xaxis_title='Spread',
        yaxis_title='Count',
        title_x=0.5,
        title_y=0.9,
        font_size=20,
    )
    return fig

def create_total_score_hist(data):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data))
    fig.update_layout(
        title='Predicted Total Score Distribution',
        xaxis_title='Total Score',
        yaxis_title='Count',
        title_x=0.5,
        title_y=0.9,
        font_size=20,
    )
    return fig

def create_feature_barchart(features, team1, data1, team2, data2):
    # TODO: x-label: feature
    # TODO: y-label: relative performance
    # TODO: title: Decisive features
    fig = go.Figure(data=[
        go.Bar(name=team1, x=features, y=data1),
        go.Bar(name=team2, x=features, y=data2),
    ])
    fig.update_layout(
        title='Team Performance',
        barmode='group',
        xaxis_title='Feature',
        yaxis_title='Relative Performance',
        title_x=0.5,
        title_font_size=40,
        yaxis_title_font_size=30,
        xaxis_title_font_size=30,
        xaxis_tickfont_size=20
    )
    return fig

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
    #TODO Package bootstrap with assets
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

plot_width = 800
plot_height = 400

# define the app layout
dashapp.layout = html.Div(
    children=[
        html.H1("Beat The Bookie", className="text-center mt-4", style={'font-size': 60}),
        dbc.Row(
            [
                dbc.Col(),
                dbc.Col(
                    dcc.Dropdown(
                        current_teams,
                        current_teams[0],
                        id="team1-dropdown",
                        className='mx-auto mt-4',
                        style={'textAlign': 'center', 'font-size': 20},
                        clearable=False,
                    ), # End Dropdown
                ),
                dbc.Col()
            ]
        ),
        dbc.Row( # Begin Score Card Row
            [
                dbc.Col(),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("", className='card-title', id='team1_card_title', style={'font-size': 40}),
                            html.P("", className='card-info', id='team1_card_text', style={'font-size': 40}),
                        ]),
                        className='card text-center',
                        id='team1-card'
                    ),
                    className='mx-auto mt-4'
                ),
                dbc.Col(
                    html.P("vs."),
                    className='my-auto',
                    id="versus-sign",
                    style={'font-size': '50px',  'text-align': 'center', 'height': '100px'}
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("", className='card-title', id='team2_card_title', style={'font-size': 40}),
                            html.P("", className='card-info', id='team2_card_text', style={'font-size': 40}),
                        ]),
                        className='card text-center',
                        id='team2-card'
                    ),
                    className='mx-auto mt-4'
                ),
                dbc.Col(),
            ]
        ), # End Score Card Row
        dbc.Row( # Begin Total Score and Spread Row
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("Predicted Total Score", className='card-title', style={'font-size': 40}),
                            html.P("", className='card-info', id='total_score_card_text', style={'font-size': 40}),
                        ]),
                        className='card text-center'
                    ),
                    className='mx-auto mt-4'
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("Predicted Spread", className='card-title', id='spread_card_title', style={'font-size': 40}),
                            html.P("", className='card-info', id='spread_card_text', style={'font-size': 40}),
                        ]),
                        className='card text-center'
                    ),
                    className='mx-auto mt-4'
                ),
            ]
        ),
        # Begin Overlay Histogram
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(id='overlap_hist'),
                    className="mx-auto mt-4",
                ),
            ]
        ),
        # Begin side by side hists
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(id='total_score_hist'),
                    className="col-md-6",
                ),
                dbc.Col(
                    dcc.Graph(id='spread_hist'),
                    className="col-md-6",
                ),
            ]
        ),
        # End side by side hists
        # Begin Decisive Features
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(id='features_barchart'),
                    className="mx-auto mt-4",
                ),
            ]
        ),
    ],
    className='container'
)

# dashapp.layout = html.Div(
#     [
#         dbc.Row(
#             html.Div(
#                 children=[
#                     html.H1(
#                         children="Beat the Bookie",
#                         style={"textAlign": "center"},
#                     ),
#                     dbc.Row(
#                         [dbc.Col(team1_dropdown)],
#                         style={
#                             "display": "flex",
#                             "justify-content": "center",
#                         },
#                     ),
#                 ]
#             )
#         ),
#         dbc.Row(
#             children=[
#                 dbc.Col(team1_card, className='mx-auto'),
#                 dbc.Col(),
#                 dbc.Col()
#             ],
#             className='mt-4'
#         ),
#         dbc.Row(
#             html.Div(
#                 children=[
#                     dbc.Row(
#                         [
#                             dcc.Graph(
#                                 id="graph-content1",
#                                 style={
#                                     "height": plot_height,
#                                     "width": plot_width,
#                                     "scale": "100%"
#                                 },
#                             ),
#                         ],
#                         style={
#                             "margin": "auto",
#                             "width": "25%",
#                         },
#                     ),
#                 ],
#             ),
#         ),
#     ],
#     className='container'
# )


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


@callback(
    [
        Output('team1_card_title','children'),
        Output('team1_card_text','children'),
        Output('team2_card_title','children'),
        Output('team2_card_text','children'),
        Output('total_score_card_text','children'),
        Output('spread_card_text','children'),
        Output('overlap_hist','figure'),
        Output('total_score_hist','figure'),
        Output('spread_hist','figure'),
        Output('features_barchart','figure'),
        Output('versus-sign','children'),

    ],
    Input("team1-dropdown", "value")
)
def update_graph(team1: str):
    # Get team 1 stats
    team1_abbrv = get_team_abbrv(team1)
    team1_stats = get_team_stats(team1_abbrv)

    # Get team 2 name
    team2_abbrv = team1_stats.opponent

    if team2_abbrv == "BYE":
        # Bye week logic
        return team1_abbrv, 0, 'None', 0, 0, 0, go.Figure(),go.Figure(),go.Figure(),go.Figure(), 'BYE Week'

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
    team1_pred_score = round(MODEL.predict(team1_input_vector)[0])
    team1_tree_preds = get_tree_predictions(MODEL, team1_input_vector)

    # Get prediction and histogram data for team 2
    team2_pred_score = round(MODEL.predict(team2_input_vector)[0])
    team2_tree_preds = get_tree_predictions(MODEL, team2_input_vector)

    # Compute total points and spread
    total_points = team1_pred_score + team2_pred_score
    spread = team1_pred_score - team2_pred_score

    # Get df.describe on histogram data
    tree_df = pd.DataFrame({'team1_score': team1_tree_preds, 'team2_score': team2_tree_preds})
    tree_df['spread'] = tree_df['team1_score'] - tree_df['team2_score']
    tree_df['total_score'] = tree_df['team1_score'] + tree_df['team2_score']

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

    ### Gathering different widget outputs ###
    overlap_hist = create_overlap_hist(team1_abbrv, team1_tree_preds, team2_abbrv, team2_tree_preds)
    spread_hist = create_spread_hist(tree_df['spread'])
    total_score_hist = create_total_score_hist(tree_df['total_score'])
    features_barchart = create_feature_barchart(important_feature_names, team1_abbrv, team1_important_feature_values.to_list(), team2_abbrv, team2_important_feature_values.to_list())

    versus_sign = 'vs.' if team1_stats.is_home else '@'

    return team1_abbrv, team1_pred_score, team2_abbrv, team2_pred_score, total_points, spread, overlap_hist, total_score_hist, spread_hist, features_barchart, versus_sign


def main():
    dashapp.run(debug=True)

if __name__ == "__main__":
    main()
