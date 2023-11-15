from flask.views import MethodView
from flask_smorest import Blueprint, abort
from Resources.schemas import ModelPredictionSchema
from Tables.model_prediction import ModelPredictionTable
from db import db
from Models.Predictions.predictor import Predictor


ModelPredictionBlueprint = Blueprint("Model Predictions", __name__, 
                                     description="API Endpoint to make NFL betting predictions and save predictions to a postgres database")


@ModelPredictionBlueprint.route("/get_all")
class ModelPredictionGetAll(MethodView):

    @ModelPredictionBlueprint.response(200, ModelPredictionSchema)
    def get(self):
        """API to get all previous predictions"""
        return ModelPredictionTable.query.all()


@ModelPredictionBlueprint.route("/delete_all")
class ModelPredictionDeleteAll(MethodView):

    @ModelPredictionBlueprint.response(200, ModelPredictionSchema)
    def delete(self):
        """API to delete all previous predictions"""
        predictions = ModelPredictionTable.query.all()
        for prediction in predictions:
            prediction_id = prediction[0].id
            ModelPredictionDeletion().delete(prediction_id)
        return {"message": "All predictions deleted."}


@ModelPredictionBlueprint.route("/delete/<string:prediction_id>")
class ModelPredictionDeletion(MethodView):

    @ModelPredictionBlueprint.response(200, ModelPredictionSchema)
    def delete(self, prediction_id):
        """API to delete a single existing prediction"""
        prediction = ModelPredictionTable.query.get_or_404(prediction_id)
        db.session.delete(prediction)
        db.session.commit()
        return {"message": "Prediction deleted."}


@ModelPredictionBlueprint.route("/predict/<int:week>/<string:home_team>/<string:away_team>/<string:favorite>/<string:given_spread>/<string:given_total>/<string:stadium>/<string:playoff>/<string:neutral_site>")
class ModelPredictionList(MethodView):

    @ModelPredictionBlueprint.response(201, ModelPredictionSchema)
    def post(self, week, home_team, away_team, favorite, given_spread, given_total, stadium, playoff, neutral_site):
        """API to make predictions"""
        try:
            # Collect provided data
            formatted_given_spread = float(given_spread)
            formatted_given_total = float(given_total)
            formatted_playoff = int(eval(playoff))
            formatted_neutral_site = int(eval(neutral_site))

            # Initialize predictor
            predictor = Predictor(week, home_team, away_team, favorite, formatted_given_spread, formatted_given_total,
                                  stadium, formatted_playoff, formatted_neutral_site)

            # Make predictions
            spread_prediction = predictor.predict_spread()
            favorite_to_cover_prediction = predictor.predict_favorite_to_cover()
            total_points_prediction = predictor.predict_total_points()
            over_to_cover_prediction = predictor.predict_over_to_cover()

        except Exception as e:
            abort(500, message=f"An error occurred while generating predictions. Ensure all variables were entered correctly; Details: {e}")

        prediction_data = {
            "home_team": home_team,
            "away_team": away_team,
            "given_spread": formatted_given_spread,
            "given_total": formatted_given_total,
            "predicted_spread": spread_prediction,
            "predicted_favorite_cover": favorite_to_cover_prediction,
            "predicted_total": total_points_prediction,
            "predicted_over_cover": over_to_cover_prediction
        }

        # Persist data to sql
        row = ModelPredictionTable(**prediction_data)

        try:
            # Delete old data
            old_prediction = ModelPredictionTable.query.all()
            if len(old_prediction) > 0:
                old_prediction_id = old_prediction[0].id
                ModelPredictionDeletion().delete(old_prediction_id)
            # Add new predictions
            db.session.add(row)
            db.session.commit()
        except Exception as e:
            abort(500, message=f"An error occurred persisting the prediction; Details: {e}")

        # Return data in json format
        return prediction_data

