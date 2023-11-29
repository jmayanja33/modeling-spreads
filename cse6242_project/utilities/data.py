import os
from datetime import datetime

import nfl_data_py as ndp
import pandas as pd

# from cse6242_project import PROJECT_ROOT

def load_weekly_stats():
    current_year = datetime.now().year
    player_stats = ndp.import_weekly_data([current_year])
    
    weekly_stats = player_stats.groupby(['recent_team', 'season', 'week']).sum().reset_index().rename(columns={'recent_team':'team','season':'year'})

    # Drop columns we don't care about
    weekly_stats = weekly_stats.drop(columns=[
        'player_id',
        'player_name',
        'player_display_name',
        'position',
        'position_group',
        'headshot_url',
        'season_type',
        'opponent_team'
    ])

    # TODO Parse nfl.xlsx for 2023 historical scoring

    # List of features we want to aggregate for training
    features = [
        # "score",
        # "other_score",
        "passing_yards",
        "passing_tds",
        "interceptions",
        "sacks",
        "sack_yards",
        "sack_fumbles",
        "sack_fumbles_lost",
        "passing_air_yards",
        "passing_yards_after_catch",
        "passing_first_downs",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "rushing_fumbles",
        "rushing_fumbles_lost",
        "rushing_first_downs",
        "receiving_fumbles",
        "receiving_fumbles_lost",
        "special_teams_tds",
    ]
    
    # Get seasonal stats for each team
    seasonal_stats_df = weekly_stats.loc[:, ['team'] + features].groupby('team', as_index=False).mean(numeric_only=True)

    return seasonal_stats_df
    
if __name__ == "__main__":
    load_weekly_stats()