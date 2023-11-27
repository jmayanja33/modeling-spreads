ABBRV2FULL = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LV": "Las Vegas Raiders",
    "LAC": "Los Angeles Chargers",
    "LA": "Los Angeles Rams",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SF": "San Francisco 49ers",
    "SEA": "Seattle Seahawks",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
}

FULL2ABBRV = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Oakland Raiders": "LV",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Diego Chargers": "LAC",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "St. Louis Rams": "LA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
    "Washington Football Team": "WAS",
    "Washington Redskins": "WAS",
}


def get_team_abbrv(full_name: str) -> str:
    """Given full team name, get team name abbreviation.

    Args
        full_name (str): Full name of team.
    Returns
        Abbreviation of team name.
    """
    if full_name not in FULL2ABBRV.keys():
        raise ValueError(
            f"Provided name '{full_name}' not legal. Must be in the following list: {','.join(FULL2ABBRV.keys())}"
        )
    else:
        return FULL2ABBRV[full_name]


def get_team_fullname(abbrv_name: str) -> str:
    """Given team name abbreviation, get full team name.

    Args
        abbrv_name (str): team name abbreviation.
    Returns
        Full team name.
    """
    if abbrv_name not in ABBRV2FULL.keys():
        raise ValueError(
            f"Provided name '{abbrv_name}' not legal. Must be in the following list: {','.join(ABBRV2FULL.keys())}"
        )
    else:
        return ABBRV2FULL[abbrv_name]
