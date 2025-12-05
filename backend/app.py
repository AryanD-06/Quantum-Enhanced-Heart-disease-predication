from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import tensorflow as tf
import os
from datetime import datetime
from bson import ObjectId

# Local imports
from database import db_instance
from auth import hash_password, verify_password, generate_token, token_required

# -------------------- APP INIT --------------------

app = Flask(__name__)
CORS(app)

model = None
scaler = None
pca = None
num_qubits = None

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model_artifacts', 'qcnn_model.keras')
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), 'model_artifacts', 'preprocessors.pkl')


# =========================================================
# LAZY LOADING (Fix for Render)
# =========================================================

def load_model_artifacts():
    """Load QCNN model + preprocessors ONLY when required."""
    global model, scaler, pca, num_qubits

    if model is not None:
        return  # already loaded

    try:
        print("🚀 Loading QCNN model...")
        model = tf.keras.models.load_model(MODEL_PATH)

        print("🚀 Loading preprocessors...")
        with open(PREPROCESSOR_PATH, 'rb') as f:
            artifacts = pickle.load(f)
            scaler = artifacts['scaler']
            pca = artifacts['pca']
            num_qubits = artifacts['num_qubits']

        print("✅ Model & preprocessors loaded successfully!")

    except Exception as e:
        print(f"❌ Error loading model artifacts: {e}")
        raise


def lazy_load():
    """Used inside routes to ensure late (safe) loading."""
    if model is None:
        load_model_artifacts()


# ======================== DATABASE INIT =========================
db_instance.connect()


# ========================== ROUTES ==============================

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Quantum Heart Disease Prediction API',
        'version': '1.0.0'
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check WITHOUT loading model."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'preprocessors_loaded': scaler is not None and pca is not None,
        'db_connected': db_instance.db is not None
    })


# ====================== Authentication ==========================

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    try:
        data = request.json
        required = ['email', 'password', 'name']

        for f in required:
            if f not in data:
                return jsonify({'error': f'Missing: {f}'}), 400

        email = data['email'].lower().strip()
        password = data['password']
        name = data['name'].strip()

        if '@' not in email:
            return jsonify({'error': 'Invalid email format'}), 400

        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400

        users = db_instance.get_collection('users')
        if users.find_one({'email': email}):
            return jsonify({'error': 'Email already exists'}), 409

        hashed = hash_password(password)
        doc = {
            'email': email,
            'password': hashed,
            'name': name,
            'created_at': datetime.utcnow()
        }

        result = users.insert_one(doc)
        token = generate_token(result.inserted_id, email)

        return jsonify({
            'message': 'Signup successful',
            'token': token,
            'user': {
                'id': str(result.inserted_id),
                'email': email,
                'name': name
            }
        }), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/auth/signin', methods=['POST'])
def signin():
    try:
        data = request.json
        email = data.get('email', '').lower().strip()
        password = data.get('password', '')

        users = db_instance.get_collection('users')
        user = users.find_one({'email': email})

        if not user or not verify_password(password, user['password']):
            return jsonify({'error': 'Invalid email or password'}), 401

        token = generate_token(user['_id'], email)

        return jsonify({
            'message': 'Login successful',
            'token': token,
            'user': {
                'id': str(user['_id']),
                'email': user['email'],
                'name': user['name']
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/auth/me', methods=['GET'])
@token_required
def get_current_user():
    try:
        users = db_instance.get_collection('users')
        user = users.find_one({'_id': ObjectId(request.user_id)})

        if not user:
            return jsonify({'error': 'User not found'}), 404

        return jsonify({
            'user': {
                'id': str(user['_id']),
                'email': user['email'],
                'name': user['name'],
                'created_at': user['created_at'].isoformat()
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ===================== PREDICTION ROUTE =========================

@app.route('/predict', methods=['POST'])
@token_required
def predict():
    try:
        lazy_load()  # <-- LOAD MODEL HERE (not at startup)

        data = request.json

        features = [
            data.get('HighBP', 0), data.get('HighChol', 0), data.get('CholCheck', 0),
            data.get('BMI', 0), data.get('Smoker', 0), data.get('Stroke', 0),
            data.get('Diabetes', 0), data.get('PhysActivity', 0), data.get('Fruits', 0),
            data.get('Veggies', 0), data.get('HvyAlcoholConsump', 0),
            data.get('AnyHealthcare', 0), data.get('NoDocbcCost', 0),
            data.get('GenHlth', 0), data.get('MentHlth', 0), data.get('PhysHlth', 0),
            data.get('DiffWalk', 0), data.get('Sex', 0), data.get('Age', 0),
            data.get('Education', 0), data.get('Income', 0)
        ]

        arr = np.array(features).reshape(1, -1)
        arr_scaled = scaler.transform(arr)
        arr_pca = pca.transform(arr_scaled)

        if arr_pca.shape[1] < num_qubits:
            pad = np.zeros((1, num_qubits - arr_pca.shape[1]))
            arr_pca = np.hstack([arr_pca, pad])

        proba = float(model.predict(arr_pca, verbose=0)[0][0])
        prediction = int(proba > 0.5)

        risk = "Low"
        if proba >= 0.6:
            risk = "High"
        elif proba >= 0.3:
            risk = "Medium"

        confidence = abs(proba - 0.5) * 2

        # Save to DB
        predictions = db_instance.get_collection('predictions')
        predictions.insert_one({
            'user_id': ObjectId(request.user_id),
            'input_data': data,
            'prediction': prediction,
            'probability': proba,
            'risk_level': risk,
            'confidence': float(confidence),
            'created_at': datetime.utcnow()
        })

        return jsonify({
            'prediction': prediction,
            'probability': proba,
            'risk_level': risk,
            'confidence': float(confidence)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ====================== Prediction History =======================

@app.route('/api/predictions/history', methods=['GET'])
@token_required
def history():
    try:
        predictions = db_instance.get_collection('predictions')

        docs = list(predictions.find({'user_id': ObjectId(request.user_id)})
                    .sort('created_at', -1).limit(50))

        for d in docs:
            d['_id'] = str(d['_id'])
            d['user_id'] = str(d['user_id'])
            d['created_at'] = d['created_at'].isoformat()

        return jsonify({'predictions': docs, 'count': len(docs)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ====================== USER STATISTICS ==========================

@app.route('/api/statistics', methods=['GET'])
@token_required
def statistics():
    try:
        predictions = db_instance.get_collection('predictions')
        docs = list(predictions.find({'user_id': ObjectId(request.user_id)}))

        total = len(docs)
        high = sum(1 for p in docs if p['risk_level'] == 'High')
        med = sum(1 for p in docs if p['risk_level'] == 'Medium')
        low = sum(1 for p in docs if p['risk_level'] == 'Low')
        avg = sum(p['probability'] for p in docs) / total if total else 0

        return jsonify({
            'total_predictions': total,
            'high_risk': high,
            'medium_risk': med,
            'low_risk': low,
            'average_risk': round(avg * 100, 1)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =================== ADMIN STATISTICS ============================

@app.route('/api/admin/statistics', methods=['GET'])
@token_required
def admin_stats():
    try:
        predictions = db_instance.get_collection('predictions')
        users = db_instance.get_collection('users')

        total_predictions = predictions.count_documents({})
        active_patients = len(predictions.distinct('user_id'))
        total_users = users.count_documents({})
        high_risk = predictions.count_documents({'risk_level': 'High'})

        from datetime import timedelta
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent = predictions.count_documents({'created_at': {'$gte': week_ago}})

        return jsonify({
            'total_predictions': total_predictions,
            'active_patients': active_patients,
            'total_users': total_users,
            'high_risk_predictions': high_risk,
            'recent_predictions_7days': recent
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =================== LOCAL ENTRYPOINT ============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
