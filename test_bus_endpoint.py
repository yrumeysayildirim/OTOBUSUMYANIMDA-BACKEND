import pytest
from fastapi.testclient import TestClient
from api import app  
import pandas as pd

client = TestClient(app)
MODEL_PKL = "regression_model.pkl"  

def test_prediction_endpoint(monkeypatch):
    
    from sklearn.ensemble import RandomForestClassifier
    import pickle
    import tempfile

    # Dummy model
    X = [[600, 100, 2], [700, 200, 2], [800, 300, 2]]
    y = [0, 1, 2]  # HIGH, LOW, MEDIUM
    model = RandomForestClassifier()
    model.fit(X, y)

    # Save model to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        pickle.dump(model, tmp)
        tmp_model_path = tmp.name

    import api as file 

    monkeypatch.setattr(file, "MODEL_PKL", tmp_model_path)
    mock_time_data = pd.DataFrame(columns=["time_slot_minutes", "day_encoded", "bus_stop_count"])
    monkeypatch.setattr(file, "time_data", mock_time_data)
    monkeypatch.setattr(file, "get_local_day", lambda: "Wednesday")
    monkeypatch.setattr(file, "noon_predict", lambda x: 150)
    monkeypatch.setattr(file, "evening_predict", lambda x: 200)

    response = client.post(
        "/474-classification-prediction",
        json={"time": "10:30", "day": "Wednesday"} 
    )

    assert response.status_code == 200
    assert "_474_predicted_density" in response.json()
    assert response.json()["_474_predicted_density"] in {"LOW", "MEDIUM", "HIGH"}
