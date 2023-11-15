from marshmallow import Schema, fields


class PlainModelPredictionSchema(Schema):
    id = fields.Int(dump_only=True)
    home_team = fields.Str(required=True)
    away_team = fields.Str(required=True)
    given_spread = fields.Float()
    given_total = fields.Float()
    predicted_spread = fields.Float()
    predicted_favorite_cover = fields.Float()
    predicted_total = fields.Float()
    predicted_over_cover = fields.Float()


class ModelPredictionSchema(PlainModelPredictionSchema):
    id = fields.Int(dump_only=True)
