# import joblib

# def load_model(model_path, preproc_path):
#     """
#     Load trained ML model and preprocessing objects
#     """
#     model = joblib.load(model_path)
#     preproc = joblib.load(preproc_path)
#     return model, preproc

import joblib

def load_model(model_path, preproc_path):
    model = joblib.load(model_path)
    preproc = joblib.load(preproc_path)
    return model, preproc