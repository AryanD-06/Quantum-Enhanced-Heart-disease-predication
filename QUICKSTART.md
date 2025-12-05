# Quick Start Guide - Quantum Heart Disease Prediction

## Prerequisites

- Python 3.8+
- Node.js 16+
- MongoDB Atlas account (already configured)

## Step 1: Generate the Model Files (PKL)

Run the QCNN training script to create the model artifacts:

```bash
python qcnn.py
```

This will:
- Train the quantum CNN model
- Create `backend/model_artifacts/` folder
- Save `qcnn_model.keras` (the trained model)
- Save `preprocessors.pkl` (scaler, PCA, config)

⏱️ This may take 10-30 minutes depending on your hardware.

## Step 2: Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

New dependencies include:
- `pymongo` - MongoDB driver
- `python-dotenv` - Environment variable management
- `bcrypt` - Password hashing
- `pyjwt` - JWT token generation

## Step 3: Configure Environment (Already Done)

The `.env` file is already created with MongoDB Atlas credentials:
- Database: `QML_heart_disease`
- Collections: `users`, `predictions`

**Important**: Keep the `.env` file secure and never commit it to version control.

## Step 4: Start the Backend Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

You should see:
- ✅ Model loaded successfully
- ✅ Preprocessors loaded successfully
- ✅ Connected to MongoDB Atlas

## Step 4: Test the Backend (Optional)

In a new terminal:

```bash
cd backend
python test_api.py
```

## Step 5: Start the Frontend

```bash
cd quantum-heart-insights
npm install
npm run dev
```

The frontend will start on `http://localhost:5173` (or similar)

## Step 6: Create Your Account

1. Navigate to `http://localhost:5173`
2. You'll be redirected to the Sign In page
3. Click "Sign up" to create a new account
4. Enter your name, email, and password
5. You'll be automatically signed in and redirected to the dashboard

## Features

### Authentication
- ✅ User registration and login
- ✅ JWT token-based authentication
- ✅ Protected routes (requires login)
- ✅ Secure password hashing with bcrypt
- ✅ User profile management

### Predictions
- ✅ Heart disease risk prediction using QCNN
- ✅ Prediction history saved to database
- ✅ View past predictions
- ✅ Risk level analysis (Low/Medium/High)

### Database
- ✅ MongoDB Atlas integration
- ✅ User data storage
- ✅ Prediction history tracking
- ✅ Secure data encryption

## API Endpoints

### Authentication
- `POST /api/auth/signup` - Register new user
- `POST /api/auth/signin` - Login user
- `GET /api/auth/me` - Get current user info

### Predictions
- `POST /predict` - Make prediction (requires authentication)
- `GET /api/predictions/history` - Get prediction history

### Health Check
- `GET /health` - Check API and database status

## Troubleshooting

### Model files not found
- Make sure you ran `python qcnn.py` first
- Check that `backend/model_artifacts/` folder exists with both files

### MongoDB connection error
- Verify your IP is whitelisted in MongoDB Atlas
- Check `.env` file has correct credentials
- Ensure internet connectivity

### Backend connection error
- Ensure backend is running on port 5000
- Check CORS is enabled (already configured)

### Frontend API errors
- Verify backend is running
- Check browser console for detailed errors
- Ensure you're signed in (token in localStorage)

### Authentication issues
- Clear browser localStorage and sign in again
- Check if JWT token hasn't expired (24 hours)
- Verify backend is connected to MongoDB

## Security Notes

- Passwords are hashed with bcrypt before storage
- JWT tokens expire after 24 hours
- All prediction endpoints require authentication
- MongoDB connection uses authentication
- Keep `.env` file secure and private

For detailed authentication setup, see `backend/AUTH_SETUP.md`
