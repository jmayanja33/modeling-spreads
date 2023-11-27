from db import db


class ModelPredictionTable(db.Model):
    __tablename__ = "modelprediction"

    id = db.Column(db.Integer, primary_key=True)
    home_team = db.Column(db.String(200), unique=True, nullable=False)
    away_team = db.Column(db.String(200), unique=True, nullable=False)
    given_spread = db.Column(db.Float(precision=1), unique=False, nullable=False)
    given_total = db.Column(db.Float(precision=1), unique=False, nullable=False)
    predicted_spread = db.Column(db.Float(precision=1), unique=False, nullable=False)
    predicted_favorite_cover = db.Column(db.Float(precision=1), unique=False, nullable=False)
    probability_predicted_team_cover = db.Column(db.Float(precision=1), unique=False, nullable=False)
    predicted_total = db.Column(db.Float(precision=1), unique=False, nullable=False)
    predicted_over_cover = db.Column(db.Float(precision=4), unique=False, nullable=False)
    probability_predicted_points_cover = db.Column(db.Float(precision=1), unique=False, nullable=False)
