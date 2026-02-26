from pymongo import MongoClient

# 1. Define connection parameters
server = "localhost"
port = 27017
dbname = "test_database"

# 2. Create the connection (MongoClient)
# In Python, this is typically done using a URI string
client = MongoClient(f"mongodb://{server}:{port}/")

# 3. Select the database
db = client[dbname]

# 4. Check if connection is successful (isopen equivalent)
try:
    # The 'ping' command is the standard way to verify an active connection
    client.admin.command('ping')
    is_open = True
except Exception:
    is_open = False

collections = db.list_collection_names()

collection = db["test_2dsphere"]
first_doc = collection.find_one()

results = collection.find({"ptime": {"$gt": 1761539784.0, "$lt": 1761539785.0}})

for doc in results:
    print(doc)

print(collections)
#print(f"Connection open: {is_open}")nts(filter={}))