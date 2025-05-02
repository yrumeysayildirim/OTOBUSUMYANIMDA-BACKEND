from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from datetime import datetime, timedelta, timezone
import random
from model_api.classification_model_api import classification_predict
from model_api.regression_model_api import regression_predict
from model_api.regression_evening_model_api import evening_predict
from model_api.regression_noon_model_api import noon_predict

app = FastAPI()
DOMAIN_NAME = "http://otobusumyanimda.com.tr"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[DOMAIN_NAME],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


time_data = pd.read_csv('classification_data_time_check.csv')

bus_prefrences = {'474':0.65, '486':0.20, '472':0.10, '477':0.05}


def get_local_day():
    utc_now = datetime.now(timezone.utc)
    local_time = utc_now + timedelta(hours=3)  # UTC+3
    return local_time.strftime('%A')


class PredictionInput(BaseModel):
    time: str  # string like "07:30"
    day: str   # string like "Monday"

class DensityPredictionInput(BaseModel):
    timeStr: str  # string like "07:30"
    day: str   # string like "Monday"

def convert_time_to_minutes(time_str):
    hours, minutes = map(int, time_str.split(":"))
    return hours * 60 + minutes


def encode_day(day_str):
    days = {
        "Monday": 1,
        "Tuesday": 3,
        "Wednesday": 4,
        "Thursday": 2,
        "Friday": 0,
        "Saturday": 5,
        "Sunday": 6
    }
    return days.get(day_str, -1)  


bus_474_weekdays = [
    '06:15', '06:30', '06:40', '06:50', '07:00', '07:10', '07:20',
    '07:30', '07:40', '07:50', '08:00', '08:15', '08:30', '08:50',
    '09:10', '09:32', '09:54', '10:16', '10:38', '11:00', '11:22',
    '11:44', '12:06', '12:28', '12:50', '13:10', '13:30', '13:50',
    '14:05', '14:20', '14:32', '14:44', '14:55', '15:05', '15:15',
    '15:25', '15:35', '15:45', '15:56', '16:08', '16:20', '16:35',
    '16:50', '17:00', '17:15', '17:30', '17:42', '17:54', '18:06',
    '18:18', '18:30', '18:45', '19:00', '19:20', '19:40', '20:00',
    '20:20', '20:45', '21:10', '21:30', '21:55', '22:30'
  ]

bus_477_weekdays = ['06:55', '07:55', '10:00', '12:00',
     '14:05', '16:05', '18:05']

bus_472_weekdays = ['06:20', '06:40', '07:20', '07:45', '08:15',
     '08:35', '09:30','10:15', '11:00', '12:00', '13:00', '13:45', '14:30',
     '15:15', '16:00', '16:45', '17:30', '18:10', '18:50', '19:30', '20:10',
     '20:40', '21:00', '21:15', '22:00']

bus_486_weekdays = ['06:15', '06:26', '06:38', '06:50', '07:02', '07:15', '07:30',
      '07:45', '08:00', '08:20', '08:40', '09:00', '09:22', '09:44',
      '10:06', '10:28', '10:50', '11:12', '11:34', '11:56', '12:18',
      '12:40', '13:00', '13:20', '13:40', '14:00', '14:15', '14:30',
      '14:45', '15:00', '15:15', '15:30', '15:45', '16:00', '16:10',
      '16:20', '16:30', '16:40', '16:50', '17:00', '17:10', '17:20',
      '17:30', '17:40', '17:50', '18:00', '18:18', '18:36', '18:54',
      '19:12', '19:30', '19:50', '20:05', '20:20', '20:35', '20:50',
      '21:05', '21:20', '21:35', '21:50', '22:05', '22:20', '22:40',
      '23:00']

bus_474_weekends = [
      '06:30', '06:50', '07:10', '07:30', '07:50', '08:10', '08:30',
            '08:50', '09:10', '09:30', '09:50', '10:10', '10:30', '10:50',
            '11:10', '11:30', '11:50', '12:10', '12:30', '12:50', '13:10',
            '13:30', '13:50', '14:10', '14:30', '14:50', '15:10', '15:30',
            '15:45', '16:00', '16:15', '16:30', '16:45', '17:00', '17:20',
            '17:40', '18:00', '18:20', '18:40', '19:00', '19:20', '19:40',
            '20:00', '20:20', '20:40', '21:00', '21:20', '21:40', '22:00'
    ]
  
bus_477_weekends = ['06:55', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00']
bus_472_weekends = ['06:30', '07:15', '07:45', '08:20', '09:00', '10:00', '11:00',
            '11:45', '12:30', '13:30', '14:30', '15:15', '16:00', '16:45',
            '17:45', '18:45', '19:45', '20:45', '21:15', '22:00']
  
bus_486_weekends = ['06:20', '06:35', '06:50', '07:05', '07:20', '07:35', '07:50',
            '08:10', '08:30', '08:50', '09:10', '09:30', '09:50', '10:10',
            '10:30', '10:50', '11:10', '11:30', '11:55', '12:20', '12:45',
            '13:10', '13:30', '13:50', '14:10', '14:30', '14:50', '15:05',
            '15:20', '15:35', '15:50', '16:10', '16:30', '16:45', '17:00',
            '17:20', '17:40', '18:05', '18:25', '18:40', '18:55', '19:10',
            '19:25', '19:40', '19:55', '20:10', '20:25', '20:40', '20:55',
            '21:10', '21:25', '21:40', '21:55', '22:10', '22:25', '22:40',
            '23:00']


@app.get('/')

def hello():

    return {'hello':'world'}


@app.get('/classification-prediction')

async def prediction():

    time = 980
    day = 0
    time_check = time_data.loc[(time_data['time_slot_minutes'] == time) & (time_data['day_encoded'] == day)]
    student_count = 78

    if not time_check.empty:

        student_count = time_check['bus_stop_count']
        student_count = int(student_count.iloc[0])
    

    density = classification_predict([time, student_count, day])

    return density


@app.get('/regression-prediction')

async def prediction():

    time = 915
    day = 0

    counts = regression_predict([time, day])
    prediction =  counts[0] 
    adjusted_prediction = counts[1]


    return f'prediction = {prediction}\nadjusted prediction = {adjusted_prediction}'


@app.get('/regression-noon-prediction')

async def prediction():

    time = 730
    day = 0

    prediction = noon_predict([time, day])

    return prediction


@app.get('/regression-evening-prediction')

async def prediction():

    # 970, 975,980, 985, 990

    time = 990
    day = 4

    prediction = evening_predict([time, day])


    return prediction




@app.post('/noon-classification-prediction')

async def prediction():
    
    time = 720  # Ideally this should come from the request, but keeping as-is for now
    day = 0

    bus_preferences = {'474': 0.65, '486': 0.20, '472': 0.10, '477': 0.05}

    # Initialize student_count
    student_count = None

    # Check if exact time match exists
    time_check = time_data.loc[(time_data['time_slot_minutes'] == time) & (time_data['day_encoded'] == day)]

    if not time_check.empty:
        student_count = int(time_check['bus_stop_count'].iloc[0])

    else:
        closest_time_threshold = 30  # ±30 minutes

        # Search for available nearby times
        available_times = time_data.loc[
            (time_data['day_encoded'] == day) &
            (time_data['time_slot_minutes'].between(time - closest_time_threshold, time + closest_time_threshold))
        ].copy()

        if not available_times.empty:
            available_times['time_diff'] = (available_times['time_slot_minutes'] - time).abs()
            closest_row = available_times.sort_values('time_diff').iloc[0]

            student_count = int(closest_row['bus_stop_count'])

            # Apply decay if necessary
            if closest_row['time_slot_minutes'] < time:
                minutes_passed = time - closest_row['time_slot_minutes']
                decay_factor = 1 - (minutes_passed / closest_time_threshold)
                decay_factor = max(decay_factor, 0.3)
                student_count = int(student_count * decay_factor)

        else:
            # If no close time exists, fallback to regression model
            student_count = noon_predict([time, day])
            student_count = int(student_count)

    # Now, split the total student count across buses based on preferences
    
    bus_predictions = []

    for bus_number, percentage in bus_preferences.items():
        bus_student_count = int(student_count * percentage)
        bus_density = classification_predict([time, bus_student_count, day])

        bus_predictions[bus_number] = {
            'predicted_student_count': bus_student_count,
            'predicted_density': bus_density
        }

    return bus_predictions

@app.post('/density-classification-prediction')

async def prediction(input: DensityPredictionInput):
    # 🌟 Convert incoming strings to usable numbers
    time = convert_time_to_minutes(input.timeStr)
    day = encode_day(input.day)
    x = datetime.now()
    dday = get_local_day()
    day = encode_day(dday)

    if day == -1:
        return {"error": "Invalid day sent!"}

    # 🛠 Now the rest of your original logic works exactly as before!
    bus_preferences = {'474': 0.65, '486': 0.20, '472': 0.10, '477': 0.05}
    student_count = None

    time_check = time_data.loc[
        (time_data['time_slot_minutes'] == time) & (time_data['day_encoded'] == day)
    ]

    if not time_check.empty:
        student_count = int(time_check['bus_stop_count'].iloc[0])
    else:
        closest_time_threshold = 30  # ±30 minutes
        available_times = time_data.loc[
            (time_data['day_encoded'] == day) &
            (time_data['time_slot_minutes'].between(time - closest_time_threshold, time + closest_time_threshold))
        ].copy()

        if not available_times.empty:
            available_times['time_diff'] = (available_times['time_slot_minutes'] - time).abs()
            closest_row = available_times.sort_values('time_diff').iloc[0]
            student_count = int(closest_row['bus_stop_count'])

            if closest_row['time_slot_minutes'] < time:
                minutes_passed = time - closest_row['time_slot_minutes']
                decay_factor = 1 - (minutes_passed / closest_time_threshold)
                decay_factor = max(decay_factor, 0.3)
                student_count = int(student_count * decay_factor)
        else:

            if ((time >= 480) and (time < 840)):
                student_count = noon_predict([time, day])
                student_count = int(student_count)

            elif ((time >= 840) and (time < 1140)):
                student_count = evening_predict([time, day])
                student_count = int(student_count)
            else:
                student_count = random.choice(range(10, 30))
                student_count = int(student_count)
    
    if ((time >= 750) and (time < 820)):
        student_count = random.choice(range(180, 300))
        student_count = int(student_count)
    # Now, split the total student count across buses based on preferences
    
    bus_student_count = int(student_count)

    # Predict density for each bus separately
    bus_density = classification_predict([time, bus_student_count, day])

    print(f'predicted_density : {bus_density}')
    return {'prediction' : bus_density}

@app.post('/474-classification-prediction')

async def prediction(input: PredictionInput):
    # 🌟 Convert incoming strings to usable numbers
    time = convert_time_to_minutes(input.time)
    day = encode_day(input.day)
    x = datetime.now()
    #dday = x.strftime("%A")
    dday = get_local_day()
    day = encode_day(dday)

    if day == -1:
        return {"error": "Invalid day sent!"}

    # 🛠 Now the rest of your original logic works exactly as before!
    bus_preferences = {'474': 0.65, '486': 0.20, '472': 0.10, '477': 0.05}
    student_count = None

    time_check = time_data.loc[
        (time_data['time_slot_minutes'] == time) & (time_data['day_encoded'] == day)
    ]

    if not time_check.empty:
        student_count = int(time_check['bus_stop_count'].iloc[0])
    else:
        closest_time_threshold = 30  # ±30 minutes
        available_times = time_data.loc[
            (time_data['day_encoded'] == day) &
            (time_data['time_slot_minutes'].between(time - closest_time_threshold, time + closest_time_threshold))
        ].copy()

        if not available_times.empty:
            available_times['time_diff'] = (available_times['time_slot_minutes'] - time).abs()
            closest_row = available_times.sort_values('time_diff').iloc[0]
            student_count = int(closest_row['bus_stop_count'])

            if closest_row['time_slot_minutes'] < time:
                minutes_passed = time - closest_row['time_slot_minutes']
                decay_factor = 1 - (minutes_passed / closest_time_threshold)
                decay_factor = max(decay_factor, 0.3)
                student_count = int(student_count * decay_factor)
        else:

            if ((time >= 480) and (time < 840)):
                student_count = noon_predict([time, day])
                student_count = int(student_count)

            elif ((time >= 840) and (time < 1140)):
                student_count = evening_predict([time, day])
                student_count = int(student_count)
            else:
                student_count = random.choice(range(10, 30))
                student_count = int(student_count)
    
    if ((time >= 750) and (time < 820)):
        student_count = random.choice(range(180, 300))
        student_count = int(student_count)
    # Now, split the total student count across buses based on preferences
    
    bus_student_count = abs(int(student_count))

    # Predict density for each bus separately
    bus_density = classification_predict([time, bus_student_count, day])


    return {'_474_predicted_density' : bus_density}


@app.post('/477-classification-prediction')

async def prediction(input: PredictionInput):
    # 🌟 Convert incoming strings to usable numbers
    time = convert_time_to_minutes(input.time)
    day = encode_day(input.day)
    x = datetime.now()
    dday = get_local_day()
    day = encode_day(dday)

    if day == -1:
        return {"error": "Invalid day sent!"}

    # 🛠 Now the rest of your original logic works exactly as before!
    bus_preferences = {'474': 0.65, '486': 0.20, '472': 0.10, '477': 0.05}
    student_count = None

    time_check = time_data.loc[
        (time_data['time_slot_minutes'] == time) & (time_data['day_encoded'] == day)
    ]

    if not time_check.empty:
        student_count = int(time_check['bus_stop_count'].iloc[0])
    else:
        closest_time_threshold = 30  # ±30 minutes
        available_times = time_data.loc[
            (time_data['day_encoded'] == day) &
            (time_data['time_slot_minutes'].between(time - closest_time_threshold, time + closest_time_threshold))
        ].copy()

        if not available_times.empty:
            available_times['time_diff'] = (available_times['time_slot_minutes'] - time).abs()
            closest_row = available_times.sort_values('time_diff').iloc[0]
            student_count = int(closest_row['bus_stop_count'])

            if closest_row['time_slot_minutes'] < time:
                minutes_passed = time - closest_row['time_slot_minutes']
                decay_factor = 1 - (minutes_passed / closest_time_threshold)
                decay_factor = max(decay_factor, 0.3)
                student_count = int(student_count * decay_factor)
        else:

            if ((time >= 480) and (time < 840)):
                student_count = noon_predict([time, day])
                student_count = int(student_count)

            elif ((time >= 840) and (time < 1140)):
                student_count = evening_predict([time, day])
                student_count = int(student_count)
            else:
                student_count = random.choice(range(10, 30))
                student_count = int(student_count)
    
    if ((time >= 750) and (time < 820)):
        student_count = random.choice(range(180, 300))
        student_count = int(student_count)
    # Now, split the total student count across buses based on preferences
    
    bus_student_count = abs(int(student_count))

    # Predict density for each bus separately
    bus_density = classification_predict([time, bus_student_count, day])


    return {'_477_predicted_density' : bus_density}

@app.post('/472-classification-prediction')

async def prediction(input: PredictionInput):
    # 🌟 Convert incoming strings to usable numbers
    time = convert_time_to_minutes(input.time)
    day = encode_day(input.day)
    x = datetime.now()
    #dday = x.strftime("%A")
    dday = get_local_day()
    day = encode_day(dday)

    if day == -1:
        return {"error": "Invalid day sent!"}

    # 🛠 Now the rest of your original logic works exactly as before!
    bus_preferences = {'474': 0.65, '486': 0.20, '472': 0.10, '477': 0.05}
    student_count = None

    time_check = time_data.loc[
        (time_data['time_slot_minutes'] == time) & (time_data['day_encoded'] == day)
    ]

    if not time_check.empty:
        student_count = int(time_check['bus_stop_count'].iloc[0])
    else:
        closest_time_threshold = 30  # ±30 minutes
        available_times = time_data.loc[
            (time_data['day_encoded'] == day) &
            (time_data['time_slot_minutes'].between(time - closest_time_threshold, time + closest_time_threshold))
        ].copy()

        if not available_times.empty:
            available_times['time_diff'] = (available_times['time_slot_minutes'] - time).abs()
            closest_row = available_times.sort_values('time_diff').iloc[0]
            student_count = int(closest_row['bus_stop_count'])

            if closest_row['time_slot_minutes'] < time:
                minutes_passed = time - closest_row['time_slot_minutes']
                decay_factor = 1 - (minutes_passed / closest_time_threshold)
                decay_factor = max(decay_factor, 0.3)
                student_count = int(student_count * decay_factor)
        else:

            if ((time >= 480) and (time < 840)):
                student_count = noon_predict([time, day])
                student_count = int(student_count)

            elif ((time >= 840) and (time < 1140)):
                student_count = evening_predict([time, day])
                student_count = int(student_count)
            else:
                student_count = random.choice(range(10, 30))
                student_count = int(student_count)
    

    if ((time >= 750) and (time < 820)):
        student_count = random.choice(range(180, 300))
        student_count = int(student_count)
    # Now, split the total student count across buses based on preferences
    
    bus_student_count = abs(int(student_count))

    # Predict density for each bus separately
    bus_density = classification_predict([time, bus_student_count, day])


    return {'_472_predicted_density' : bus_density}

@app.post('/486-classification-prediction')

async def prediction(input: PredictionInput):
    # 🌟 Convert incoming strings to usable numbers
    time = convert_time_to_minutes(input.time)
    day = encode_day(input.day)
    x = datetime.now()
    #dday = x.strftime("%A")
    dday = get_local_day()
    day = encode_day(dday)

    if day == -1:
        return {"error": "Invalid day sent!"}

    # 🛠 Now the rest of your original logic works exactly as before!
    bus_preferences = {'474': 0.65, '486': 0.20, '472': 0.10, '477': 0.05}
    student_count = None

    time_check = time_data.loc[
        (time_data['time_slot_minutes'] == time) & (time_data['day_encoded'] == day)
    ]

    if not time_check.empty:
        student_count = int(time_check['bus_stop_count'].iloc[0])
    else:
        closest_time_threshold = 30  # ±30 minutes
        available_times = time_data.loc[
            (time_data['day_encoded'] == day) &
            (time_data['time_slot_minutes'].between(time - closest_time_threshold, time + closest_time_threshold))
        ].copy()

        if not available_times.empty:
            available_times['time_diff'] = (available_times['time_slot_minutes'] - time).abs()
            closest_row = available_times.sort_values('time_diff').iloc[0]
            student_count = int(closest_row['bus_stop_count'])

            if closest_row['time_slot_minutes'] < time:
                minutes_passed = time - closest_row['time_slot_minutes']
                decay_factor = 1 - (minutes_passed / closest_time_threshold)
                decay_factor = max(decay_factor, 0.3)
                student_count = int(student_count * decay_factor)
        else:

            if ((time >= 480) and (time < 840)):
                student_count = noon_predict([time, day])
                student_count = int(student_count)

            elif ((time >= 840) and (time < 1140)):
                student_count = evening_predict([time, day])
                student_count = int(student_count)
            else:
                student_count = random.choice(range(10, 30))
                student_count = int(student_count)
    
    if ((time >= 750) and (time < 820)):
        student_count = random.choice(range(180, 300))
        student_count = int(student_count)

    # Now, split the total student count across buses based on preferences
    
    bus_student_count = abs(int(student_count))

    # Predict density for each bus separately
    bus_density = classification_predict([time, bus_student_count, day])


    return {'_486_predicted_density' : bus_density}

@app.post('/sd-pie')

async def prediction(input: PredictionInput):
    # 🌟 Convert incoming strings to usable numbers
    time = convert_time_to_minutes(input.time)
    day = encode_day(input.day)
    x = datetime.now()
    dday = get_local_day()
    #dday = x.strftime("%A")
    day = encode_day(dday)

    if day == -1:
        return {"error": "Invalid day sent!"}

    # 🛠 Now the rest of your original logic works exactly as before!
    bus_preferences = {'474': 0.65, '486': 0.20, '472': 0.10, '477': 0.05}
    student_count = None

    time_check = time_data.loc[
        (time_data['time_slot_minutes'] == time) & (time_data['day_encoded'] == day)
    ]

    if not time_check.empty:
        student_count = int(time_check['bus_stop_count'].iloc[0])
    else:
        closest_time_threshold = 30  # ±30 minutes
        available_times = time_data.loc[
            (time_data['day_encoded'] == day) &
            (time_data['time_slot_minutes'].between(time - closest_time_threshold, time + closest_time_threshold))
        ].copy()

        if not available_times.empty:
            available_times['time_diff'] = (available_times['time_slot_minutes'] - time).abs()
            closest_row = available_times.sort_values('time_diff').iloc[0]
            student_count = int(closest_row['bus_stop_count'])

            if closest_row['time_slot_minutes'] < time:
                minutes_passed = time - closest_row['time_slot_minutes']
                decay_factor = 1 - (minutes_passed / closest_time_threshold)
                decay_factor = max(decay_factor, 0.3)
                student_count = int(student_count * decay_factor)
        else:

            if ((time >= 480) and (time < 840)):
                student_count = noon_predict([time, day])
                student_count = int(student_count)

            elif ((time >= 840) and (time < 1140)):
                student_count = evening_predict([time, day])
                student_count = int(student_count)
            else:
                student_count = random.choice(range(10, 30))
                student_count = int(student_count)
    
    if ((time >= 750) and (time < 820)):
        student_count = random.choice(range(180, 300))
        student_count = int(student_count)
    # Now, split the total student count across buses based on preferences
    
    bus_student_count = abs(int(student_count))
    print(bus_student_count)

    return {'student count' : bus_student_count}

@app.post('/stop-density-pie')

async def prediction(input: PredictionInput):
    # 🌟 Convert incoming strings to usable numbers
    time = convert_time_to_minutes(input.time)
    day = encode_day(input.day)
    x = datetime.now()
    dday = get_local_day()
    #dday = x.strftime("%A")
    day = encode_day(dday)

    std_counts = []
    weekdays = [bus_474_weekdays, bus_472_weekdays, bus_477_weekdays, bus_486_weekdays]


    if day == -1:
        return {"error": "Invalid day sent!"}
    
    if (day != 5 or day != 6):

        for bus_day in weekdays:

            for bus in bus_day:

                time = convert_time_to_minutes(bus)

                # 🛠 Now the rest of your original logic works exactly as before!
                bus_preferences = {'474': 0.65, '486': 0.20, '472': 0.10, '477': 0.05}
                student_count = None

                time_check = time_data.loc[
                    (time_data['time_slot_minutes'] == time) & (time_data['day_encoded'] == day)
                ]

                if not time_check.empty:
                    student_count = int(time_check['bus_stop_count'].iloc[0])
                else:
                    closest_time_threshold = 30  # ±30 minutes
                    available_times = time_data.loc[
                        (time_data['day_encoded'] == day) &
                        (time_data['time_slot_minutes'].between(time - closest_time_threshold, time + closest_time_threshold))
                    ].copy()

                    if not available_times.empty:
                        available_times['time_diff'] = (available_times['time_slot_minutes'] - time).abs()
                        closest_row = available_times.sort_values('time_diff').iloc[0]
                        student_count = int(closest_row['bus_stop_count'])

                        if closest_row['time_slot_minutes'] < time:
                            minutes_passed = time - closest_row['time_slot_minutes']
                            decay_factor = 1 - (minutes_passed / closest_time_threshold)
                            decay_factor = max(decay_factor, 0.3)
                            student_count = int(student_count * decay_factor)
                    else:

                        if ((time >= 480) and (time < 840)):
                            student_count = noon_predict([time, day])
                            student_count = int(student_count)

                        elif ((time >= 840) and (time < 1140)):
                            student_count = evening_predict([time, day])
                            student_count = int(student_count)
                        else:
                            student_count = random.choice(range(10, 30))
                            student_count = int(student_count)
                

                # Now, split the total student count across buses based on preferences
                
                bus_student_count = abs(int(student_count))
                std_counts.append(bus_student_count)
    max_count = max(std_counts)
    print(max_count)


    return {'maximum' : max_count}

@app.post('/evening-classification-prediction')

async def prediction():
    time = 720  # Ideally this should come from the request, but keeping as-is for now
    day = 0

    # Initialize student_count
    student_count = None

    # Check if exact time match exists
    time_check = time_data.loc[(time_data['time_slot_minutes'] == time) & (time_data['day_encoded'] == day)]

    if not time_check.empty:
        student_count = int(time_check['bus_stop_count'].iloc[0])

    else:
        closest_time_threshold = 30  # ±30 minutes

        # Search for available nearby times
        available_times = time_data.loc[
            (time_data['day_encoded'] == day) &
            (time_data['time_slot_minutes'].between(time - closest_time_threshold, time + closest_time_threshold))
        ].copy()

        if not available_times.empty:
            available_times['time_diff'] = (available_times['time_slot_minutes'] - time).abs()
            closest_row = available_times.sort_values('time_diff').iloc[0]

            student_count = int(closest_row['bus_stop_count'])

            # Apply decay if necessary
            if closest_row['time_slot_minutes'] < time:
                minutes_passed = time - closest_row['time_slot_minutes']
                decay_factor = 1 - (minutes_passed / closest_time_threshold)
                decay_factor = max(decay_factor, 0.3)
                student_count = student_count * decay_factor

        else:
            # If no close time exists, fallback to regression model
            student_count = evening_predict([time, day])
            student_count = int(student_count)

    # Now that student_count is set correctly, predict the density
    density = classification_predict([time, student_count, day])

    return {
        'predicted_student_count': int(student_count),
        'predicted_density': density
    }
