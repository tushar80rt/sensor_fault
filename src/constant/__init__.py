import os

# ---------------------------------------------------------------------------
# Cloud / Storage
# ---------------------------------------------------------------------------
AWS_S3_BUCKET_NAME    = os.getenv("AWS_S3_BUCKET_NAME", "water-fault")

# ---------------------------------------------------------------------------
# MongoDB
# ---------------------------------------------------------------------------
MONGO_DATABASE_NAME   = "sensor"
MONGO_COLLECTION_NAME = "waferfault"
MONGO_DB_URL          = os.getenv("MONGO_DB_URL")

if not MONGO_DB_URL:
    raise EnvironmentError(
        "MONGO_DB_URL environment variable is not set. "
        "Please add it to your .env file or export it in your shell before running the app."
    )

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
TARGET_COLUMNS      = "quality"
MODEL_FILE_NAME     = "model"
MODEL_FILE_EXTENSION = ".pkl"

# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------
artifact_folder = "artifacts"