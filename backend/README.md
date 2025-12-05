# Quantum Heart Disease Prediction Backend

## Setup Instructions

### 1. Generate Model Files
First, run the QCNN training script to generate the model artifacts:

```bash
python qcnn.py
```

This will create:
- `backend/model_artifacts/qcnn_model.keras` - The trained model
- `backend/model_artifacts/preprocessors.pkl` - Scaler, PCA, and config

### 2. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 3. Run the Server
```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### Health Check
```
GET /health
```

### Prediction
```
POST /predict
Content-Type: application/json

{
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
```

Response:
```json
{
  "prediction": 1,
  "probability": 0.75,
  "risk_level": "High",
  "confidence": 0.5
}
```
