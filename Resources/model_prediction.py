import uuid
from flask import request
from flask.views import MethodView
from flask_smorest import Blueprint, abort
from schemas import ModelPredictionSchema
from Tables import ModelPredictionTable
from db import db
from sqlalchemy.exc import SQLAlchemyError


ModelPredictionBlueprint = Blueprint("Model Predictions", __name__, 
                                     description="API Endpoints to update/query prediction data")


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


@ModelPredictionBlueprint.route("/predict")
class ModelPredictionList(MethodView):

    @ModelPredictionBlueprint.response(200, ModelPredictionSchema(many=True))
    def get(self):
        """Method to get all existing prediction data"""
        return ModelPredictionTable.query.all()

    @ModelPredictionBlueprint.arguments(ModelPredictionSchema)
    @ModelPredictionBlueprint.response(201, ModelPredictionSchema)
    def post(self, prediction_data):
        prediction = ModelPredictionTable(**prediction_data)
        try:
            db.session.add(prediction)
            db.session.commit()
        except SQLAlchemyError:
            abort(500, message="An error occurred while inserting the item.")

        return prediction

    def predict_spread(self):
        """Function to predict and persist spread"""
        raise NotImplementedError

    def predict_total(self):
        """Function to predict and persist points total"""
        raise NotImplementedError
