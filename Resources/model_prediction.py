import uuid
from flask import request
from flask.views import MethodView
from flask_smorest import Blueprint, abort
from schemas import ModelPredictionSchema
from Tables import ModelPredictionTable
from db import db
from sqlalchemy.exc import SQLAlchemyError
from Models import Predictor


ModelPredictionBlueprint = Blueprint("Model Predictions", __name__, 
                                     description="API Endpoints to make predictions/query prediction data")


@ModelPredictionBlueprint.route("/predict/<string:prediction_id>")
class ModelPrediction(MethodView):

    @ModelPredictionBlueprint.response(200, ModelPredictionSchema)
    def get(self, prediction_id):
        """Method to get data from an existing prediction"""
        prediction = ModelPredictionTable.query.get_or_404(prediction_id)
        return prediction

    @ModelPredictionBlueprint.response(200, ModelPredictionSchema)
    def delete(self, prediction_id):
        """Method to delete an existing prediction"""
        prediction = ModelPredictionTable.query.get_or_404(prediction_id)
        db.session.delete(prediction)
        db.session.commit()
        return {"message": "Prediction deleted."}


@ModelPredictionBlueprint.route("/predict/<int:week>/<string:home_team>/<string:away_team>/<string:favorite>/<float:given_spread>/<float:given_total>/<string:stadium>/<boolean:playoff>/<boolean:neutral_site>")
class ModelPredictionList(MethodView):

    @ModelPredictionBlueprint.arguments(ModelPredictionSchema)
    @ModelPredictionBlueprint.response(201, ModelPredictionSchema)
    def post(self):
        # Collect provided data
        week = request.args.get("week")
        home_team = request.args.get("home_team")
        away_team = request.args.get("away_team")
        favorite = request.args.get("favorite")
        given_spread = request.args.get("given_spread")
        given_total = request.args.get("given_total")
        stadium = request.args.get("stadium")
        if request.args.get("playoff"):
            playoff = 1
        else:
            playoff = 0
        if request.args.get("neutral_site"):
            neutral_site = 1
        else:
            neutral_site = 0

        try:
            # Initialize predictor
            predictor = Predictor(week, home_team, away_team, favorite, given_spread, given_total, stadium, playoff,
                                  neutral_site)

            # Make predictions
            spread_prediction = predictor.predict_spread()
            favorite_to_cover_prediction = predictor.predict_favorite_to_cover()
            total_points_prediction = predictor.predict_total_points()
            over_to_cover_prediction = predictor.predict_over_to_cover()
        except Exception as e:
            abort(500, message=f"An error occurred while inserting the prediction; Details: {e}")

        prediction_data = {
            "home_team": home_team,
            "away_team": away_team,
            "given_spread": given_spread,
            "given_total": given_total,
            "predicted_spread": spread_prediction,
            "predicted_favorite_cover": favorite_to_cover_prediction,
            "predicted_total": total_points_prediction,
            "predicted_over_cover": over_to_cover_prediction
        }

        # Persist data to sql
        row = ModelPredictionTable(**prediction_data)
        try:
            db.session.delete()
            db.session.add(row)
            db.session.commit()
        except SQLAlchemyError:
            abort(500, message="An error occurred while inserting the prediction.")

        # Return data in json format
        return prediction_data

