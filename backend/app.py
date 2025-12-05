from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import tensorflow as tf
import os
from datetime import datetime
from bson import ObjectId
from database import db_instance
from auth import hash_password, verify_password, generate_token, token_required

app = Flask(__name__)
CORS(app)  # enable CORS right after app creation

# ----------------- MODEL & PREPROCESSORS -----------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model_artifacts', 'qcnn_model.keras')
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), 'model_artifacts', 'preprocessors.pkl')

model = None
scaler = None
pca = None
num_qubits = None


def load_model_artifacts():
    """Load the trained model and preprocessors"""
    global model, scaler, pca, num_qubits

    try:
        # Load Keras model
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully")

        # Load preprocessors
        with open(PREPROCESSOR_PATH, 'rb') as f:
            artifacts = pickle.load(f)
            scaler = artifacts['scaler']
            pca = artifacts['pca']
            num_qubits = artifacts['num_qubits']
        print("✅ Preprocessors loaded successfully")

    except Exception as e:
        print(f"❌ Error loading model artifacts: {e}")
        raise


# Load artifacts on startup
load_model_artifacts()

# Connect to database (will log error if connection fails)
db_instance.connect()

# ----------------- ROUTES -----------------


@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        'message': 'Quantum Heart Disease Prediction API',
        'version': '1.0.0',
        'endpoints': {
            'health': '/health',
            'predict': '/predict (POST)'
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'preprocessors_loaded': scaler is not None and pca is not None,
        'database_connected': db_instance.db is not None
    })


# ==================== Authentication Routes ====================

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    """User registration endpoint"""
    try:
        data = request.json

        # Validate required fields
        required_fields = ['email', 'password', 'name']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        email = data['email'].lower().strip()
        password = data['password']
        name = data['name'].strip()

        # Validate email format
        if '@' not in email:
            return jsonify({'error': 'Invalid email format'}), 400

        # Validate password length
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400

        # Check if user already exists
        users_collection = db_instance.get_collection('users')
        existing_user = users_collection.find_one({'email': email})

        if existing_user:
            return jsonify({'error': 'User with this email already exists'}), 409

        # Hash password
        hashed_password = hash_password(password)

        # Create user document
        user_doc = {
            'email': email,
            'password': hashed_password,
            'name': name,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }

        # Insert user
        result = users_collection.insert_one(user_doc)
        user_id = result.inserted_id

        # Generate token
        token = generate_token(user_id, email)

        return jsonify({
            'message': 'User registered successfully',
            'token': token,
            'user': {
                'id': str(user_id),
                'email': email,
                'name': name
            }
        }), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/auth/signin', methods=['POST'])
def signin():
    """User login endpoint"""
    try:
        data = request.json

        # Validate required fields
        if 'email' not in data or 'password' not in data:
            return jsonify({'error': 'Email and password are required'}), 400

        email = data['email'].lower().strip()
        password = data['password']

        # Find user
        users_collection = db_instance.get_collection('users')
        user = users_collection.find_one({'email': email})

        if not user:
            return jsonify({'error': 'Invalid email or password'}), 401

        # Verify password
        if not verify_password(password, user['password']):
            return jsonify({'error': 'Invalid email or password'}), 401

        # Generate token
        token = generate_token(user['_id'], email)

        return jsonify({
            'message': 'Login successful',
            'token': token,
            'user': {
                'id': str(user['_id']),
                'email': user['email'],
                'name': user['name']
            }
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/auth/me', methods=['GET'])
@token_required
def get_current_user():
    """Get current user information"""
    try:
        users_collection = db_instance.get_collection('users')
        user = users_collection.find_one({'_id': ObjectId(request.user_id)})

        if not user:
            return jsonify({'error': 'User not found'}), 404

        return jsonify({
            'user': {
                'id': str(user['_id']),
                'email': user['email'],
                'name': user['name'],
                'created_at': user['created_at'].isoformat()
            }
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
@token_required
def predict():
    """Prediction endpoint (requires authentication)"""
    try:
        data = request.json

        # Extract features in the correct order
        features = [
            data.get('HighBP', 0),
            data.get('HighChol', 0),
            data.get('CholCheck', 0),
            data.get('BMI', 0),
            data.get('Smoker', 0),
            data.get('Stroke', 0),
            data.get('Diabetes', 0),
            data.get('PhysActivity', 0),
            data.get('Fruits', 0),
            data.get('Veggies', 0),
            data.get('HvyAlcoholConsump', 0),
            data.get('AnyHealthcare', 0),
            data.get('NoDocbcCost', 0),
            data.get('GenHlth', 0),
            data.get('MentHlth', 0),
            data.get('PhysHlth', 0),
            data.get('DiffWalk', 0),
            data.get('Sex', 0),
            data.get('Age', 0),
            data.get('Education', 0),
            data.get('Income', 0)
        ]

        # Convert to numpy array
        input_data = np.array(features).reshape(1, -1)

        # Preprocess: Scale
        input_scaled = scaler.transform(input_data)

        # Preprocess: PCA
        input_pca = pca.transform(input_scaled)

        # Pad if necessary
        if input_pca.shape[1] < num_qubits:
            padding = np.zeros((input_pca.shape[0], num_qubits - input_pca.shape[1]))
            input_pca = np.hstack([input_pca, padding])

        # Make prediction
        prediction_proba = model.predict(input_pca, verbose=0)[0][0]
        prediction = int(prediction_proba > 0.5)

        # Calculate risk level
        if prediction_proba < 0.3:
            risk_level = 'Low'
        elif prediction_proba < 0.6:
            risk_level = 'Medium'
        else:
            risk_level = 'High'

        # Save prediction to database
        predictions_collection = db_instance.get_collection('predictions')
        prediction_doc = {
            'user_id': ObjectId(request.user_id),
            'input_data': data,
            'prediction': prediction,
            'probability': float(prediction_proba),
            'risk_level': risk_level,
            'confidence': float(abs(prediction_proba - 0.5) * 2),
            'created_at': datetime.utcnow()
        }
        predictions_collection.insert_one(prediction_doc)

        return jsonify({
            'prediction': prediction,
            'probability': float(prediction_proba),
            'risk_level': risk_level,
            'confidence': float(abs(prediction_proba - 0.5) * 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/predictions/history', methods=['GET'])
@token_required
def get_prediction_history():
    """Get user's prediction history"""
    try:
        predictions_collection = db_instance.get_collection('predictions')
        predictions = list(predictions_collection.find(
            {'user_id': ObjectId(request.user_id)}
        ).sort('created_at', -1).limit(50))

        # Convert ObjectId to string for JSON serialization
        for pred in predictions:
            pred['_id'] = str(pred['_id'])
            pred['user_id'] = str(pred['user_id'])
            pred['created_at'] = pred['created_at'].isoformat()

        return jsonify({
            'predictions': predictions,
            'count': len(predictions)
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/statistics', methods=['GET'])
@token_required
def get_statistics():
    """Get overall statistics for the current user"""
    try:
        predictions_collection = db_instance.get_collection('predictions')

        # Get user's predictions
        user_predictions = list(predictions_collection.find({'user_id': ObjectId(request.user_id)}))

        # Calculate statistics
        total_predictions = len(user_predictions)
        high_risk_count = sum(1 for p in user_predictions if p.get('risk_level') == 'High')
        medium_risk_count = sum(1 for p in user_predictions if p.get('risk_level') == 'Medium')
        low_risk_count = sum(1 for p in user_predictions if p.get('risk_level') == 'Low')

        # Calculate average risk
        if total_predictions > 0:
            avg_probability = sum(p.get('probability', 0) for p in user_predictions) / total_predictions
        else:
            avg_probability = 0

        return jsonify({
            'total_predictions': total_predictions,
            'high_risk': high_risk_count,
            'medium_risk': medium_risk_count,
            'low_risk': low_risk_count,
            'average_risk': round(avg_probability * 100, 1)
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/statistics', methods=['GET'])
@token_required
def get_admin_statistics():
    """Get overall system statistics (all users)"""
    try:
        predictions_collection = db_instance.get_collection('predictions')
        users_collection = db_instance.get_collection('users')

        # Total predictions across all users
        total_predictions = predictions_collection.count_documents({})

        # Total active patients (users who have made at least one prediction)
        active_patients = len(predictions_collection.distinct('user_id'))

        # Total registered users
        total_users = users_collection.count_documents({})

        # High risk predictions
        high_risk_count = predictions_collection.count_documents({'risk_level': 'High'})

        # Recent predictions (last 7 days)
        from datetime import timedelta
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        recent_predictions = predictions_collection.count_documents({
            'created_at': {'$gte': seven_days_ago}
        })

        return jsonify({
            'total_predictions': total_predictions,
            'active_patients': active_patients,
            'total_users': total_users,
            'high_risk_predictions': high_risk_count,
            'recent_predictions_7days': recent_predictions
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# -------- Local dev entrypoint (ignored by gunicorn on Render) --------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


