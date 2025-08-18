import os
from pymongo import MongoClient

def get_client():
    uri = os.getenv("MONGO_URI")
    return MongoClient(uri)

def get_orders_unified_col():
    client = get_client()
    db = client[os.getenv("DB_NAME", "report_db")]
    return db[os.getenv("COLL_NAME", "orders_unified")]
