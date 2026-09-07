import os

from google.cloud import storage

BUCKET_NAME = os.getenv("BUCKET_NAME")


def _parse_gs_uri(gs_uri: str):
    #split a gs://bucket/path/to/blob URI into (bucket, blob_path)
    if not gs_uri.startswith("gs://"):
        raise ValueError("GCS URI must start with gs://")
    #everything after gs://, split once on the first / -> [bucket, blob]
    parts = gs_uri[len("gs://"):].split("/", 1)
    if len(parts) < 2 or not parts[1]:
        raise ValueError("GCS URI must contain a blob path")
    return parts[0], parts[1]


def download_from_gcs(local_path: str, gs_uri: str) -> str:
    bucket_name, blob_path = _parse_gs_uri(gs_uri)
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_path)
    blob.download_to_filename(local_path)
    return local_path


def upload_file(local_path: str, dest_blob: str) -> str:
    #upload local_path to gs://BUCKET_NAME/dest_blob
    if not BUCKET_NAME:
        raise RuntimeError("BUCKET_NAME environment variable is not set")
    client = storage.Client()
    blob = client.bucket(BUCKET_NAME).blob(dest_blob)
    blob.upload_from_filename(local_path)
    return f"gs://{BUCKET_NAME}/{dest_blob}"
