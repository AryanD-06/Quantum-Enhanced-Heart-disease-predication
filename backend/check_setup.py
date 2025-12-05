import os
import sys

def check_setup():
    """Check if all required files exist"""
    print("🔍 Checking backend setup...\n")
    
    checks = {
        "Model file": "model_artifacts/qcnn_model.keras",
        "Preprocessors file": "model_artifacts/preprocessors.pkl",
        "App file": "app.py",
        "Requirements file": "requirements.txt"
    }
    
    all_good = True
    
    for name, path in checks.items():
        full_path = os.path.join(os.path.dirname(__file__), path)
        exists = os.path.exists(full_path)
        status = "✅" if exists else "❌"
        print(f"{status} {name}: {path}")
        if not exists:
            all_good = False
    
    print("\n" + "="*50)
    
    if all_good:
        print("✅ All files present! You can start the server with:")
        print("   python app.py")
    else:
        print("❌ Missing files detected!")
        print("\n💡 To generate model files, run:")
        print("   python qcnn.py")
        print("\n   (from the root directory)")
    
    print("="*50)
    
    return all_good

if __name__ == "__main__":
    success = check_setup()
    sys.exit(0 if success else 1)
