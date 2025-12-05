"""
Test MongoDB Atlas connection
"""
from database import db_instance

def test_connection():
    print("Testing MongoDB Atlas connection...")
    
    # Connect to database
    if db_instance.connect():
        print("✅ Successfully connected to MongoDB Atlas!")
        
        # Test collections
        try:
            users_collection = db_instance.get_collection('users')
            predictions_collection = db_instance.get_collection('predictions')
            
            # Count documents
            user_count = users_collection.count_documents({})
            prediction_count = predictions_collection.count_documents({})
            
            print(f"✅ Users collection: {user_count} documents")
            print(f"✅ Predictions collection: {prediction_count} documents")
            
            # List all collections
            collections = db_instance.db.list_collection_names()
            print(f"✅ Available collections: {collections}")
            
            db_instance.close()
            print("✅ Connection closed successfully")
            
        except Exception as e:
            print(f"❌ Error accessing collections: {e}")
    else:
        print("❌ Failed to connect to MongoDB Atlas")
        print("\nTroubleshooting:")
        print("1. Check if your IP address is whitelisted in MongoDB Atlas")
        print("2. Verify the connection string in .env file")
        print("3. Ensure you have internet connectivity")

if __name__ == "__main__":
    test_connection()
