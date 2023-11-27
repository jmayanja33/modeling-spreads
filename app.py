import os
from flask import Flask
from flask_smorest import Api
from db import db
from Resources.model_prediction import ModelPredictionBlueprint


def create_app():
    app = Flask(__name__)

    app.config["PROPAGATE_EXCEPTIONS"] = True
    app.config["API_TITLE"] = "Team 176 Project REST API"
    app.config["API_VERSION"] = "v1"
    app.config["OPENAPI_VERSION"] = "3.0.3"
    app.config["OPENAPI_URL_PREFIX"] = "/"
    app.config["OPENAPI_SWAGGER_UI_PATH"] = "/swagger-ui"
    app.config["OPENAPI_SWAGGER_UI_URL"] = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/"
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("POSTGRES_URL") # "sqlite:///data.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATION"] = False
    db.init_app(app)

    api = Api(app)

    with app.app_context():
        db.create_all()

    api.register_blueprint(ModelPredictionBlueprint)

    return app


# if __name__ == '__main__':
#     app = create_app()
#     app.run()
