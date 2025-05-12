import pickle
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor

MODEL_PKL= os.path.join('model_api','models','regression_noon_model.pkl')

def classification_predict(input_data):

    model = pickle.load(open(MODEL_PKL, 'rb'))

    df = pd.DataFrame([input_data], columns=["time_slot_minutes", "bus_stop_count", "day_encoded"])

    prediction = model.predict(df)

    # LOW = 1, MEDIUM = 2, HIGH = 0
    
    match prediction:

        case 0:
            prediction = 'HIGH'
        case 1:
            prediction = 'LOW'
        case 2:
            prediction = 'MEDIUM'

    return prediction


def evening_predict(input_data):

    model = pickle.load(open(MODEL_PKL, 'rb'))

    df = pd.DataFrame([input_data], columns=["time_slot_minutes", "day_encoded"])

    prediction = model.predict(df)
    
    return int(prediction)


def noon_predict(input_data):

    model = pickle.load(open(MODEL_PKL, 'rb'))

    df = pd.DataFrame([input_data], columns=["time_slot_minutes", "day_encoded"])

    prediction = model.predict(df)
    

    return int(prediction)

# tests


def test_classification_predict(tmp_path, monkeypatch):
    X = [[10, 1, 0], [20, 2, 1], [30, 3, 2]]
    y = [0, 1, 2]  # HIGH, LOW, MEDIUM

    model = RandomForestClassifier()
    model.fit(X, y)

    model_path = tmp_path / "clf_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    monkeypatch.setitem(globals(), "MODEL_PKL", str(model_path))

    pred = classification_predict([15, 2, 1])
    assert pred in {"HIGH", "LOW", "MEDIUM"}


def test_evening_predict(tmp_path, monkeypatch):
    X = [[10, 0], [20, 1], [30, 2]]
    y = [100, 200, 300]

    model = GradientBoostingRegressor()
    model.fit(X, y)

    model_path = tmp_path / "reg_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    monkeypatch.setitem(globals(), "MODEL_PKL", str(model_path))

    pred = evening_predict([25, 1])
    assert isinstance(pred, int)


def test_noon_predict(tmp_path, monkeypatch):
    X = [[5, 0], [15, 1], [25, 2]]
    y = [50, 150, 250]

    model = GradientBoostingRegressor()
    model.fit(X, y)

    model_path = tmp_path / "reg_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    monkeypatch.setitem(globals(), "MODEL_PKL", str(model_path))

    pred = noon_predict([20, 1])
    assert isinstance(pred, int)
