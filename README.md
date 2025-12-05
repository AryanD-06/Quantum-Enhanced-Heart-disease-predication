# Quantum Heart Insights 🫀✨

A quantum-inspired machine learning application for heart disease risk prediction with secure user authentication and cloud database integration.

![Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Node](https://img.shields.io/badge/node-16+-green.svg)
![MongoDB](https://img.shields.io/badge/database-MongoDB%20Atlas-brightgreen.svg)

## 🌟 Features

### Core Functionality
- 🧠 **Quantum CNN Model** - Advanced quantum-inspired neural network for heart disease prediction
- 📊 **Risk Assessment** - Comprehensive analysis with Low/Medium/High risk levels
- 📈 **Confidence Scoring** - Model confidence metrics for each prediction
- 🎯 **High Accuracy** - Trained on BRFSS 2015 health indicators dataset

### Authentication & Security
- 🔐 **User Authentication** - Secure sign up and sign in with JWT tokens
- 🔒 **Password Security** - Bcrypt hashing with salt
- 🛡️ **Protected Routes** - All predictions require authentication
- ⏰ **Token Expiration** - 24-hour JWT token validity
- 👤 **User Profiles** - Personal account management

### Data Management
- ☁️ **Cloud Database** - MongoDB Atlas integration
- 📝 **Prediction History** - All predictions saved to user account
- 🔍 **Data Tracking** - View past predictions and trends
- 💾 **Secure Storage** - Encrypted data transmission and storage

### User Interface
- 🎨 **Modern Design** - Beautiful gradient UI with Tailwind CSS
- 📱 **Responsive** - Works on desktop, tablet, and mobile
- ⚡ **Fast** - Optimized with Vite and React
- 🎭 **Intuitive** - Easy-to-use interface for all users

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- MongoDB Atlas account

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd quantum-heart-insights
```

2. **Set up MongoDB Atlas**
   - Follow instructions in `backend/MONGODB_SETUP.md`
   - Update `backend/.env` with your cluster URL

3. **Install backend dependencies**
```bash
cd backend
pip install -r requirements.txt
```

4. **Test MongoDB connection**
```bash
python test_mongodb.py
```

5. **Start backend server**
```bash
python app.py
```

6. **Install frontend dependencies** (in new terminal)
```bash
cd quantum-heart-insights
npm install
```

7. **Start frontend**
```bash
npm run dev
```

8. **Open browser**
   - Navigate to `http://localhost:5173`
   - Sign up for a new account
   - Start making predictions!

For detailed setup instructions, see `SETUP_INSTRUCTIONS.md`

## 📚 Documentation

- **[Setup Instructions](SETUP_INSTRUCTIONS.md)** - Complete setup guide
- **[Setup Checklist](SETUP_CHECKLIST.md)** - Step-by-step checklist
- **[Quick Start](QUICKSTART.md)** - Quick start guide
- **[Authentication Summary](AUTHENTICATION_SUMMARY.md)** - Authentication implementation details
- **[System Architecture](SYSTEM_ARCHITECTURE.md)** - System design and architecture
- **[MongoDB Setup](backend/MONGODB_SETUP.md)** - MongoDB Atlas configuration
- **[Auth Setup](backend/AUTH_SETUP.md)** - Authentication setup guide

## 🏗️ Architecture

```
┌─────────────┐      HTTP/JWT      ┌─────────────┐      PyMongo      ┌─────────────┐
│   React     │ ◄─────────────────► │   Flask     │ ◄────────────────► │  MongoDB    │
│  Frontend   │                     │   Backend   │                    │   Atlas     │
│  (Port 5173)│                     │  (Port 5000)│                    │   (Cloud)   │
└─────────────┘                     └─────────────┘                    └─────────────┘
     │                                     │                                   │
     │                                     │                                   │
  • Sign In/Up                        • Authentication                    • users
  • Protected Routes                  • JWT Tokens                        • predictions
  • User Profile                      • QCNN Model
  • Predictions                       • Risk Assessment
```

## 🛠️ Technology Stack

### Frontend
- **React 18** - UI framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Shadcn/ui** - UI components
- **React Router** - Navigation
- **TanStack Query** - Data fetching
- **Vite** - Build tool

### Backend
- **Python 3.8+** - Programming language
- **Flask** - Web framework
- **TensorFlow** - ML framework
- **PyMongo** - MongoDB driver
- **bcrypt** - Password hashing
- **PyJWT** - JWT tokens

### Database
- **MongoDB Atlas** - Cloud database
- **Collections:**
  - `users` - User accounts
  - `predictions` - Prediction history

### Machine Learning
- **TensorFlow/Keras** - Neural networks
- **Scikit-learn** - Preprocessing
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation

## 📊 Database Schema

### Users Collection
```javascript
{
  _id: ObjectId,
  email: String (unique),
  password: String (hashed),
  name: String,
  created_at: DateTime,
  updated_at: DateTime
}
```

### Predictions Collection
```javascript
{
  _id: ObjectId,
  user_id: ObjectId,
  input_data: {
    HighBP: Number,
    HighChol: Number,
    BMI: Number,
    // ... 21 health metrics
  },
  prediction: Number (0 or 1),
  probability: Number,
  risk_level: String,
  confidence: Number,
  created_at: DateTime
}
```

## 🔐 Security Features

1. **Password Security**
   - Bcrypt hashing with salt
   - Minimum 6 characters
   - Never stored in plain text

2. **Token Security**
   - JWT with HS256 algorithm
   - 24-hour expiration
   - Secure token storage

3. **Route Protection**
   - Frontend route guards
   - Backend authentication middleware
   - Automatic redirect to sign in

4. **Database Security**
   - MongoDB Atlas authentication
   - IP whitelist
   - Encrypted connections

## 📁 Project Structure

```
.
├── backend/
│   ├── app.py                    # Main Flask application
│   ├── database.py               # MongoDB connection
│   ├── auth.py                   # Authentication utilities
│   ├── .env                      # Environment variables
│   ├── requirements.txt          # Python dependencies
│   └── model_artifacts/          # ML model files
│
├── quantum-heart-insights/
│   └── src/
│       ├── contexts/             # React contexts
│       ├── pages/                # Page components
│       ├── components/           # Reusable components
│       └── api/                  # API client
│
├── SETUP_INSTRUCTIONS.md         # Complete setup guide
├── SETUP_CHECKLIST.md           # Setup checklist
├── AUTHENTICATION_SUMMARY.md     # Auth implementation
├── SYSTEM_ARCHITECTURE.md        # System architecture
└── README.md                     # This file
```

## 🎯 API Endpoints

### Authentication
- `POST /api/auth/signup` - Register new user
- `POST /api/auth/signin` - Login user
- `GET /api/auth/me` - Get current user

### Predictions
- `POST /predict` - Make prediction (requires auth)
- `GET /api/predictions/history` - Get prediction history (requires auth)

### Health Check
- `GET /health` - Check API and database status

## 🧪 Testing

### Test MongoDB Connection
```bash
cd backend
python test_mongodb.py
```

### Test API Endpoints
```bash
cd backend
python test_api.py
```

### Manual Testing
1. Sign up for a new account
2. Sign in with credentials
3. Make a prediction
4. View prediction history
5. Check MongoDB Atlas for data
6. Sign out and sign in again

## 🐛 Troubleshooting

### MongoDB Connection Issues
- Verify cluster URL in `.env`
- Check IP whitelist in MongoDB Atlas
- Ensure internet connectivity

### Backend Issues
- Check if port 5000 is available
- Verify all dependencies installed
- Check model files exist

### Frontend Issues
- Clear browser cache and localStorage
- Check if backend is running
- Verify API URL is correct

For detailed troubleshooting, see `SETUP_INSTRUCTIONS.md`

## 🚀 Deployment

### Frontend (Vercel/Netlify)
1. Build: `npm run build`
2. Deploy `dist` folder
3. Set environment variables

### Backend (Heroku/AWS/DigitalOcean)
1. Use Gunicorn for production
2. Set environment variables
3. Configure HTTPS
4. Set up monitoring

### Database (MongoDB Atlas)
1. Use production cluster
2. Enable automated backups
3. Configure IP whitelist
4. Set up monitoring

## 📈 Future Enhancements

- [ ] Password reset functionality
- [ ] Email verification
- [ ] Two-factor authentication
- [ ] Social login (Google, GitHub)
- [ ] Profile editing
- [ ] Export prediction history
- [ ] Advanced analytics dashboard
- [ ] Mobile app
- [ ] API rate limiting
- [ ] Comprehensive logging

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- BRFSS 2015 dataset for training data
- TensorFlow team for ML framework
- MongoDB Atlas for database hosting
- React and Flask communities

## 📞 Support

For issues or questions:
1. Check the documentation files
2. Review troubleshooting sections
3. Check MongoDB Atlas dashboard
4. Review browser console and backend logs

## 🎉 Success Indicators

✅ MongoDB Atlas connected  
✅ Backend running without errors  
✅ Frontend accessible  
✅ User registration working  
✅ Authentication functional  
✅ Predictions being made  
✅ Data saved to database  
✅ Protected routes working  

---

**Built with ❤️ using Quantum-Inspired Machine Learning**

For detailed setup instructions, start with `SETUP_INSTRUCTIONS.md` or use `SETUP_CHECKLIST.md` for a step-by-step guide.
