import os

#the API modules read configuration from the environment at import time; give
#the test session a complete, deterministic set before anything imports them
os.environ.setdefault("API_TOKENS", "tok_test:alice")
os.environ.setdefault("BUCKET_NAME", "test-bucket")
os.environ.setdefault("PROJECT_ID", "test-project")
os.environ.setdefault("REGION", "us-central1")
os.environ.setdefault(
    "VERTEX_CONTAINER_URI",
    "us-central1-docker.pkg.dev/test-project/repo/yolo-trainer:latest",
)
