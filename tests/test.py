import pytest
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import os

# Fixture to create a mock model and scaler
@pytest.fixture(scope="module")
def mock_model_and_scaler():
    # Create a simple mock model
    model = LinearRegression()
    # Fit with dummy data to initialize coefficients
    model.fit([[0.5, 0.2], [0.6, 0.3]], [1.0, 1.2])

    # Create a scaler fitted with dummy data
    scaler = StandardScaler()
    scaler.fit([[0.5, 0.2], [0.6, 0.3], [0.7, 0.4]])

    return model, scaler

def test_model_prediction_valid_input(mock_model_and_scaler):
    model, scaler = mock_model_and_scaler
    # Test input: humidity and wind_speed (raw values, will be scaled)
    input_data = pd.DataFrame([[0.5, 0.2]], columns=["humidity", "wind_speed"])

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Check that prediction is a float and within a reasonable range
    assert isinstance(prediction, float), f"Expected float, got {type(prediction)}"
    assert -3 < prediction < 3, f"Prediction {prediction} out of expected range"
    print("Test 1 (Valid input): Passed")

def test_model_prediction_edge_case(mock_model_and_scaler):
    model, scaler = mock_model_and_scaler
    # Test edge case: extreme but plausible humidity and wind_speed
    input_data = pd.DataFrame([[1.0, 0.0]], columns=["humidity", "wind_speed"])

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Check that prediction is still reasonable
    assert isinstance(prediction, float), f"Expected float, got {type(prediction)}"
    assert -5 < prediction < 5, f"Prediction {prediction} out of expected range"
    print("Test 2 (Edge case): Passed")

def test_model_prediction_invalid_input(mock_model_and_scaler):
    model, scaler = mock_model_and_scaler
    # Test invalid input: non-numeric values
    input_data = pd.DataFrame([["invalid", "invalid"]], columns=["humidity", "wind_speed"])

    # Attempt to scale (should raise an error)
    with pytest.raises(ValueError):
        scaler.transform(input_data)
    print("Test 3 (Invalid input): Passed")

if __name__ == "__main__":
    os.makedirs("tests", exist_ok=True)
    pytest.main(["-v", __file__])