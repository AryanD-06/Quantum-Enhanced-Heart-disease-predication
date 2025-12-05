import requests
import json

# Test data
test_patient = {
    "HighBP": 1,
    "HighChol": 1,
    "CholCheck": 1,
    "BMI": 28.5,
    "Smoker": 0,
    "Stroke": 0,
    "Diabetes": 0,
    "PhysActivity": 1,
    "Fruits": 1,
    "Veggies": 1,
    "HvyAlcoholConsump": 0,
    "AnyHealthcare": 1,
    "NoDocbcCost": 0,
    "GenHlth": 3,
    "MentHlth": 5,
    "PhysHlth": 10,
    "DiffWalk": 0,
    "Sex": 1,
    "Age": 9,
    "Education": 5,
    "Income": 6
}

# Test health endpoint
print("Testing health endpoint...")
response = requests.get('http://localhost:5000/health')
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}\n")

# Test prediction endpoint
print("Testing prediction endpoint...")
response = requests.post(
    'http://localhost:5000/predict',
    json=test_patient,
    headers={'Content-Type': 'application/json'}
)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")
