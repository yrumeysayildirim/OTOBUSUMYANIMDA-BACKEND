import pickle
import os
import numpy as np
import pandas as pd

MODEL_PKL= os.path.join('model_api','models','regression_evening_model.pkl')

def load_model(filename = MODEL_PKL):

    with open(filename, 'rb') as f:
        model = pickle.load(f)

    return model

def correct_peak_prediction_single(prediction, time_slot_minutes):

    """
    895 -120
    920 + 60
    925 + 50
    980 + 150
    985 - 60
    1030 + 20
    1070 + 20
    1080 + 20

    based on error chart
    """
    if time_slot_minutes in range(890, 900):      # ~14:55 → overpredicting
        return prediction - 120
    elif time_slot_minutes in range(910, 921):    # ~15:20  → underpredicting
        return prediction + 60
    elif time_slot_minutes in range(921, 930):    # ~15:25 → underpredicting
        return prediction + 50
    elif time_slot_minutes in range(970, 982):    # ~16:20 → underpredicting
        return prediction + 150
    elif time_slot_minutes in range(982, 990):    # ~16:25 → overpredicting
        return prediction - 60
    elif time_slot_minutes in range(1020, 1035):    # ~17:10 → underpredicting
        return prediction + 20
    elif time_slot_minutes in range(1060, 1075):    # ~17:50 → underpredicting
        return prediction + 20
    elif time_slot_minutes in range(1076, 1085):    # ~18:00 → underpredicting
        return prediction + 20
    else:
        return prediction
    
def evening_predict(input_data):

    model = pickle.load(open(MODEL_PKL, 'rb'))

    df = pd.DataFrame([input_data], columns=["time_slot_minutes", "day_encoded"])

    prediction = model.predict(df)

    #prediction = correct_peak_prediction_single(prediction, input_data[0])
    

    return int(prediction)


time = 925
day = 0
"""

"""


