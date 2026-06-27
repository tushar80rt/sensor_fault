from pymongo.mongo_client import MongoClient
import pandas as pd
import json
import os

# ---------------------------------------------------------------------------
# Load MongoDB URI from environment variable
# Set MONGO_DB_URL in your shell or in a .env file before running this script
# ---------------------------------------------------------------------------
uri = os.getenv("MONGO_DB_URL")

if not uri:
    raise EnvironmentError(
        "MONGO_DB_URL environment variable is not set.\n"
        "Run: set MONGO_DB_URL=mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/  (Windows)\n"
        "  or export MONGO_DB_URL=...  (Linux/macOS)"
    )

# Create a new client and connect to the server
client = MongoClient(uri)

# Database and collection names
DATABASE_NAME   = "sensor"
COLLECTION_NAME = "waferfault"

# ---------------------------------------------------------------------------
# Load the wafer CSV — path is relative to the project root
# ---------------------------------------------------------------------------
CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "notebooks",
    "wafer_23012020_041211.csv"
)

df = pd.read_csv(CSV_PATH)
df = df.drop("Unnamed: 0", axis=1)

json_record = list(json.loads(df.T.to_json()).values())

client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)
print(f"Successfully inserted {len(json_record)} records into {DATABASE_NAME}.{COLLECTION_NAME}")