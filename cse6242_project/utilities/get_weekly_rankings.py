import os
from typing import Dict, Any
from datetime import datetime, timedelta

import pandas as pd
import nfl_data_py as ndp

from cse6242_project import PROJECT_ROOT


ranked_columns = [
    'passing_yards',
    'passing_tds',
    'interceptions',
    'sacks',
    'sack_yards',
    'sack_fumbles',
    'sack_fumbles_lost',
    'passing_air_yards',
    'passing_yards_after_catch',
    'passing_first_downs',
    'passing_epa',
    'carries',
    'rushing_yards',
    'rushing_tds',
    'rushing_fumbles',
    'rushing_fumbles_lost',
    'rushing_first_downs',
    'rushing_epa',
    'receiving_fumbles',
    'receiving_fumbles_lost',
    'special_teams_tds'
]

class WeeklyRanking:
    _RANK_COLS = ranked_columns
    def __init__(self) -> None:
        self.last_update = datetime(1900, 1, 1).date()
        self._team_name_map = self._get_team_names()
        self._get_data()

    @property
    def current_year(self) -> datetime:
        return datetime.now().year

    @property
    def today(self) -> datetime:
        return datetime.now().date()

    def _get_team_names(self) -> Dict[str, str]:
        df = pd.read_csv(os.path.join(
            PROJECT_ROOT,
            'data',
            'nfl_teams.csv'
        ))
        return {k: v for k, v in zip(df.team_name, df.team_id)}

    def _get_data(self) -> pd.DataFrame:
        if (self.today - self.last_update) > timedelta(days=1):
            data = ndp.import_weekly_data([self.current_year])
            self._ranked_data = data.loc[
                :, self._RANK_COLS + ['recent_team']
            ].groupby('recent_team').sum().rank(
                method='dense', ascending=False
            )
            self._ranked_data.index.name = 'recent_team'
            self._ranked_data  = self._ranked_data.reset_index()
            self._ranked_data = self._ranked_data.rename(
                columns={
                    col: 'ranked_' + col for col in self._RANK_COLS
                }
            )
            self.last_update = self.today
        return self._ranked_data

    def get_rankings(self, team_name: str) -> Dict[str, Any]:
        if team_name not in self._team_name_map:
            raise ValueError(
                f'Team name "{team_name}" not in list of current teams: {self._team_name_map}'
            )
        else:
            team_abbrev = self._team_name_map[team_name]
        data = self._get_data()
        return data[data.recent_team == team_abbrev].loc[:, ['ranked_' + col for col in self._RANK_COLS]]
