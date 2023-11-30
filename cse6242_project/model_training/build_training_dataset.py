import os
import pandas as pd

from cse6242_project import PROJECT_ROOT

# Read in expanded data for home/away and scores
df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "expanded_data.csv"))

# Select columns we care about
df = df[['year', 'week', 'home_team_abbrv', 'away_team_abbrv', 'home_score', 'away_score']]

# Pull weekly player stats from git repo
# For current season - https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_2023.parquet
player_stats = pd.read_parquet('https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats.parquet', engine='auto')

# Group by team, season, and week to get the sum of stats
weekly_stats = player_stats.groupby(['recent_team', 'season', 'week']).sum().reset_index().rename(columns={'recent_team':'team','season':'year'})

# Create copies of base dataframe to build home and away records
df_home = df.copy()
df_home = df_home.rename(columns={'home_team_abbrv':'team','away_team_abbrv':'other_team','home_score':'score','away_score':'other_score'})[['year','team','week','other_team','score','other_score']]
df_home['is_home'] = True

df_away = df.copy()
df_away = df_away.rename(columns={'home_team_abbrv':'other_team','away_team_abbrv':'team','away_score':'score','home_score':'other_score'})[['year','team','week','other_team','score','other_score']]
df_away['is_home'] = False

df = pd.concat([df_home, df_away], ignore_index=True)

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

# Join the base dataframe with the weekly stats on year, team, and week
df = pd.merge(df, weekly_stats, how='inner', on=['year','team','week'])

# Sort the data by year, then team, then week
df = df.sort_values(by=['year','team','week']).reset_index(drop=True)

# Dump an intermediate csv
df.to_csv('seasonal_raw.csv',index=False)

# List of features we want to aggregate for training
features = [
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

# For each feature, calculate the sum and mean per team per season
for feature in features:

    # Here, we are grouping by year (season) and team, then computing the sum and mean of the feature as the season progresses
    # df[f'{feature}_sum'] = df.groupby(['year','team'])[feature].apply(lambda x: x.expanding().sum().shift()).reset_index(drop=True)
    df[f'{feature}_mean'] = df.groupby(['year','team'])[feature].apply(lambda x: x.expanding().mean().shift()).reset_index(drop=True)

# Dump an intermediate csv
df.to_csv('seasonal_aggregated.csv', index=False)

# Build list of columns we want to include in our training dataset
base_cols = ['year','team','week','other_team','score','is_home']
# sum_cols = [feature + '_sum' for feature in features]
mean_cols = [feature + '_mean' for feature in features]

# Get dataframe based on columns we want to train on
slim_df = df[base_cols + mean_cols]

# Merge with self to get Other team stats appended to the right of the Team stats
df_merge = pd.merge(slim_df, slim_df, left_on=['year','other_team','week'], right_on=['year','team','week'], suffixes=['','_other'])

# Filter to the columns we want for Other team stats
# sum_cols_other = [col + '_other' for col in sum_cols]
mean_cols_other = [col + '_other' for col in mean_cols]

cols = base_cols + mean_cols + mean_cols_other
final_df = df_merge[cols]

# drop NAN rows (week 1 has no previous stats for the season)
final_df = final_df.dropna()

# Dump final training dataframe
final_df.to_csv('training_data_2.csv', index=False)

print()