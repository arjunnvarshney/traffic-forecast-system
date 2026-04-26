from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert "MLOps Traffic Engine" in response.text

def test_predict_endpoint_invalid_hour():
    # Hour must be between 0 and 23
    response = client.post("/predict", json={"hour": 25, "temperature": 290.0, "rain": 0.0})
    assert response.status_code == 422 # Pydantic validation error

def test_predict_endpoint_negative_rain():
    # Rain must be non-negative
    response = client.post("/predict", json={"hour": 15, "temperature": 290.0, "rain": -5.0})
    assert response.status_code == 422 

def test_predict_endpoint_invalid_type():
    # Invalid datatype
    response = client.post("/predict", json={"hour": "invalid", "temperature": 290.0, "rain": 0.0})
    assert response.status_code == 422

def test_predict_endpoint_valid_request():
    # Valid data
    response = client.post("/predict", json={"hour": 15, "temperature": 290.0, "rain": 0.0})
    # Might return 503 if model isn't trained in the test environment
    assert response.status_code in [200, 503]

def test_predict_endpoint_boundary_hour():
    # Hour 0 (midnight) should succeed
    response = client.post("/predict", json={"hour": 0, "temperature": 290.0, "rain": 0.0})
    assert response.status_code in [200, 503]
    
    # Hour 23 (11pm) should succeed
    response = client.post("/predict", json={"hour": 23, "temperature": 290.0, "rain": 0.0})
    assert response.status_code in [200, 503]
