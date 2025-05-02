import pickle
import os
import numpy as np
import pandas as pd

MODEL_PKL= os.path.join('model_api','models','regression_model.pkl')
"""
def load_model(filename = MODEL_PKL):

    with open(filename, 'rb') as f:
        model = pickle.load(f)

    return model
"""
def correct_peak_prediction_single(prediction, time_slot_minutes):
    if time_slot_minutes in range(615, 625):      # ~10:15 AM → underpredicting
        return prediction + 120
    elif time_slot_minutes in range(660, 670):    # ~11:00 AM → underpredicting
        return prediction + 150
    elif time_slot_minutes in range(720, 730):    # ~12:00 PM → overpredicting
        return prediction - 75
    elif time_slot_minutes in range(810, 820):    # ~3:00 PM → overpredicting
        return prediction - 100
    elif time_slot_minutes in range(890, 900):    # ~4:00 PM → underpredicting
        return prediction - 200
    else:
        return prediction

def regression_predict(input_data):

    model = pickle.load(open(MODEL_PKL, 'rb'))

    df = pd.DataFrame([input_data], columns=["time_slot_minutes", "day_encoded"])

    prediction = model.predict(df)

    new_prediction = correct_peak_prediction_single(prediction, input_data[0])
    

    return [prediction, new_prediction]


time = 720
day = 1



#model = load_model(MODEL_PKL)
#prediction = regression_predict(model, [time, day])

#corrected_prediction = correct_peak_prediction_single(prediction, time)

#print(f'prediction = {int(prediction[0])}\ncorrected prediction = {int(corrected_prediction[0])}')


