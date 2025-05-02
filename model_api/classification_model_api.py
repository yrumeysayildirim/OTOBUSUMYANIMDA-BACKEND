import pickle
import os
import numpy as np
import pandas as pd

MODEL_PKL= os.path.join('model_api','models','classification_model.pkl')
"""
def load_model(filename = MODEL_PKL):

    with open(filename, 'rb') as f:
        model = pickle.load(f)

    return model
"""
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


time = 915
student_count = 78
day = 0



#model = load_model(MODEL_PKL)
#prediction = classification_predict([time, student_count, day])
#print(prediction)
