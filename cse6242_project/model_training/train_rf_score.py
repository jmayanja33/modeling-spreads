"""Build placeholder model."""
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor


import joblib

from cse6242_project import PROJECT_ROOT

expanded_data_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "expanded_data.csv"))

# create seasonal team rankings
ranking_features = [
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
    "passing_epa",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_fumbles",
    "rushing_fumbles_lost",
    "rushing_first_downs",
    "rushing_epa",
    "receiving_fumbles",
    "receiving_fumbles_lost",
    "special_teams_tds",
]

home_columns_to_rank = {"home_team_" + f: f for f in ranking_features}
home_columns_to_rank.update({"home_score": "score"})
home_extra_required_cols = ["year", "week", "home_team_name"]

away_columns_to_rank = {"away_team_" + f: f for f in ranking_features}
away_columns_to_rank.update({"away_score": "score"})
away_extra_required_cols = ["year", "week", "away_team_name"]

home_temp_df = (
    expanded_data_df[(expanded_data_df.week.astype(str) != "1")]
    .loc[:, home_extra_required_cols + [*home_columns_to_rank.keys()]]
    .reset_index()
    .rename(columns=home_columns_to_rank)
    .rename(columns={"home_team_name": "team_name", "index": "original_index"})
)
home_temp_df.insert(2, "is_home", 1)

away_temp_df = (
    expanded_data_df[(expanded_data_df.week.astype(str) != "1")]
    .loc[:, away_extra_required_cols + [*away_columns_to_rank.keys()]]
    .reset_index()
    .rename(columns=away_columns_to_rank)
    .rename(columns={"away_team_name": "team_name", "index": "original_index"})
)

away_temp_df.insert(2, "is_home", 0)

temp_df = pd.concat([home_temp_df, away_temp_df])
temp_df["week"] = pd.to_numeric(temp_df.week, "coerce")
temp_df = temp_df.dropna().sort_values(["year", "week"]).reset_index(drop=True)

for feature in ranking_features:
    temp_df["ranked_" + feature] = temp_df.groupby(["year", "week"])[feature].rank(
        "dense"
    )

feature_columns = ["ranked_" + feature for feature in ranking_features]  # + [
feature_df = temp_df.loc[:, feature_columns]
score_df = temp_df.loc[:, "score"]

X_train, X_test, y_train, y_test = train_test_split(
    feature_df, score_df, test_size=0.20, random_state=1
)

params = {"bootstrap": True, "max_depth": 5, "max_features": 1.0, "n_estimators": 100}
rfr = RandomForestRegressor(**params)
rfr.fit(X_train, y_train)
test_predict = rfr.predict(X_test)
score = r2_score(y_test, test_predict)

joblib.dump(rfr, os.path.join(PROJECT_ROOT, "models", "rf_regressor_score.pkl"))
