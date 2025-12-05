"""
Quick backend connectivity check
Tests if Flask can start and model files are loaded
"""
import os
import sys

print("=" * 60)
print("BACKEND CONNECTIVITY CHECK")
print("=" * 60)

# Check 1: Python version
print("\n1. Python Version:")
print(f"   ✅ Python {sys.version.split()[0]}")

# Check 2: Required packages
print("\n2. Checking Required Packages:")
required_packages = [
    'flask',
    'flask_cors',
    'numpy',
    'tensorflow',
    'sklearn',
    'pymongo',
    'bcrypt',
    'jwt',
    'dotenv'
]

missing_packages = []
for package in required_packages:
    try:
        if package == 'sklearn':
            __import__('sklearn')
        elif package == 'jwt':
            __import__('jwt')
        elif package == 'dotenv':
            __import__('dotenv')
        else:
            __import__(package)
        print(f"   ✅ {package}")
    except ImportError:
        print(f"   ❌ {package} - NOT INSTALLED")
        missing_packages.append(package)

if missing_packages:
    print(f"\n   ⚠️  Missing packages: {', '.join(missing_packages)}")
    print("   Run: pip install -r requirements.txt")

# Check 3: Model files
print("\n3. Checking Model Files:")
model_path = os.path.join(os.path.dirname(__file__), 'model_artifacts', 'qcnn_model.keras')
preprocessor_path = os.path.join(os.path.dirname(__file__), 'model_artifacts', 'preprocessors.pkl')

if os.path.exists(model_path):
    print(f"   ✅ qcnn_model.keras found")
else:
    print(f"   ❌ qcnn_model.keras NOT FOUND")
    print(f"   Run: python qcnn.py (from root directory)")

if os.path.exists(preprocessor_path):
    print(f"   ✅ preprocessors.pkl found")
else:
    print(f"   ❌ preprocessors.pkl NOT FOUND")
    print(f"   Run: python qcnn.py (from root directory)")

# Check 4: Environment file
print("\n4. Checking Environment Configuration:")
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    print(f"   ✅ .env file found")
    
    # Check if MongoDB URL is configured
    with open(env_path, 'r') as f:
        env_content = f.read()
        if '<your-cluster-url>' in env_content:
            print(f"   ⚠️  MongoDB URL not configured (still has placeholder)")
            print(f"   See GET_MONGODB_URL.md for instructions")
        else:
            print(f"   ✅ MongoDB URL appears to be configured")
else:
    print(f"   ❌ .env file NOT FOUND")

# Check 5: Port availability
print("\n5. Checking Port 5000:")
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result = sock.connect_ex(('localhost', 5000))
if result == 0:
    print(f"   ⚠️  Port 5000 is already in use")
    print(f"   Stop the existing process or use a different port")
else:
    print(f"   ✅ Port 5000 is available")
sock.close()

# Check 6: Try to import Flask app modules
print("\n6. Checking Backend Modules:")
try:
    from database import db_instance
    print(f"   ✅ database.py can be imported")
except Exception as e:
    print(f"   ❌ database.py import failed: {e}")

try:
    from auth import hash_password, generate_token
    print(f"   ✅ auth.py can be imported")
except Exception as e:
    print(f"   ❌ auth.py import failed: {e}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

if not missing_packages and os.path.exists(model_path) and os.path.exists(preprocessor_path):
    print("✅ Backend is ready to start!")
    print("\nNext steps:")
    print("1. Update MongoDB URL in backend/.env (see GET_MONGODB_URL.md)")
    print("2. Run: python app.py")
else:
    print("⚠️  Backend needs configuration")
    print("\nRequired actions:")
    if missing_packages:
        print("- Install missing packages: pip install -r requirements.txt")
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        print("- Generate model files: python qcnn.py")
    print("- Update MongoDB URL in backend/.env")

print("=" * 60)
