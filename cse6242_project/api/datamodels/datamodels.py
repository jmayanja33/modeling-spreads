from pydantic import BaseModel


class BettingLineInput(BaseModel):
    pass


class ScoreModelInput(BaseModel):
    ranked_passing_yards: float
    ranked_passing_tds: float
    ranked_interceptions: float
    ranked_sacks: float
    ranked_sack_yards: float
    ranked_sack_fumbles: float
    ranked_sack_fumbles_lost: float
    ranked_passing_air_yards: float
    ranked_passing_yards_after_catch: float
    ranked_passing_first_downs: float
    ranked_passing_epa: float
    ranked_carries: float
    ranked_rushing_yards: float
    ranked_rushing_tds: float
    ranked_rushing_fumbles: float
    ranked_rushing_fumbles_lost: float
    ranked_rushing_first_downs: float
    ranked_rushing_epa: float
    ranked_receiving_fumbles: float
    ranked_receiving_fumbles_lost: float
    ranked_special_teams_tds: float


class ScoreInput(BaseModel):
    team_name: str
