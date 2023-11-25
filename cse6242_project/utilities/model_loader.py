import os
import joblib

from cse6242_project import PROJECT_ROOT


def load_model(model_name: str):
    model_path = os.path.join(PROJECT_ROOT, 'models', model_name)
    if not os.path.isfile(model_path):
        raise FileNotFoundError('Could not find model:', model_path)
    else:
        model = joblib.load(model_path)
    return model
