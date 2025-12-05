from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import os
from dotenv import load_dotenv

load_dotenv()

class Database:
    def __init__(self):
        self.client = None
        self.db = None
        
    def connect(self):
        """Connect to MongoDB Atlas"""
        try:
            mongodb_uri = os.getenv('MONGODB_URI')
            database_name = os.getenv('DATABASE_NAME', 'QML_heart_disease')
            
            self.client = MongoClient(mongodb_uri)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[database_name]
            print(f"✅ Connected to MongoDB Atlas - Database: {database_name}")
            return True
        except ConnectionFailure as e:
            print(f"❌ Failed to connect to MongoDB: {e}")
            return False
        except Exception as e:
            print(f"❌ Error connecting to database: {e}")
            return False
    
    def get_collection(self, collection_name):
        """Get a collection from the database"""
        if self.db is None:
            raise Exception("Database not connected")
        return self.db[collection_name]
    
    def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            print("✅ Database connection closed")

# Global database instance
db_instance = Database()
