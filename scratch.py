import nfl_data_py as nfl

group_cols = ["recent_team", "season_type", "week"]
keep_cols = ["recent_team", "season_type", "week", "completions", "attempts", "passing_yards", "passing_tds",
             "interceptions", "sacks", "sack_yards", "sack_fumbles", "sack_fumbles_lost", "passing_air_yards",
             "passing_yards_after_catch", "passing_first_downs", "passing_epa", "passing_2pt_conversions", "pacr",
             "carries", "rushing_yards", "rushing_tds", "rushing_fumbles", "rushing_fumbles_lost",
             "rushing_first_downs", "rushing_epa", "rushing_2pt_conversions", "receiving_fumbles",
             "receiving_fumbles_lost", "special_teams_tds", "fantasy_points", "fantasy_points_ppr"]

weekly_data = nfl.import_weekly_data([2022])
test_df = weekly_data.groupby(["recent_team", "season_type", "week"], as_index=False).sum()
test_df = test_df[keep_cols]
pass