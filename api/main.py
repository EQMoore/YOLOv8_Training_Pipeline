import io
import os
import re
import shutil
import tempfile

from fastapi import Depends, FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from google.cloud import storage

import gcs_util
from auth import current_user

app = FastAPI()

#http response codes: https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status

#resource names go into GCS object paths, so keep them to a safe character set
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
#job_id is "yolo-train-{user_id}-{uuid}"; longer than a resource name, and it
#is interpolated into a Vertex list filter, so keep it to a safe character set
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_ARTIFACTS = {"final_model.pt", "final_model.onnx", "final_model.quant.onnx"}


def _validate_name(value: str, field: str) -> str:
    if not _NAME_RE.match(value):
        raise HTTPException(status_code=400, detail=f"Invalid {field}")
    return value


@app.post("/train_yolo", status_code=202)
def train_yolo(
    dataset: UploadFile,
    model: str = Form(...),
    epochs: int = Form(10),
    batch: int = Form(16),
    arch: str = Form("yolov8n"),
    user_id: str = Depends(current_user),
):
    if not dataset.filename:
        raise HTTPException(status_code=400, detail="No dataset uploaded")
    _validate_name(model, "model")
    _validate_name(arch, "arch")

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_file:
        temp_path = temp_file.name
        shutil.copyfileobj(dataset.file, temp_file)

    blob_path = f"{user_id}/{model}.zip"
    try:
        if gcs_util.check_gcs_unique_name(f"{user_id}/{model}"):
            raise HTTPException(status_code=409, detail="Model name already in use")

        gcs_uri = gcs_util.upload_to_gcs(temp_path, blob_path)
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

    try:
        job_id = gcs_util.submit_training_job(gcs_uri, user_id, model, epochs, batch, arch)
    except Exception as exc:
        #the dataset is in GCS but no job was created; remove it so this model
        #name isn't permanently blocked, then surface the failure
        try:
            gcs_util.delete_from_gcs(blob_path)
        except Exception:
            pass
        raise HTTPException(
            status_code=502, detail="Failed to submit training job"
        ) from exc

    return {"job_id": job_id, "user_id": user_id, "model": model, "status": "submitted"}


@app.get("/job_status")
def job_status(job_id: str, user_id: str = Depends(current_user)):
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=400, detail="Invalid job_id")
    status = gcs_util.get_training_status(job_id, user_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


@app.get("/get_models")
def get_models(user_id: str = Depends(current_user)):
    return gcs_util.get_user_models(user_id)


@app.get("/download_model")
def download_model(model_name: str, user_id: str = Depends(current_user)):
    _validate_name(model_name, "model_name")
    return gcs_util.get_user_models(f"{user_id}/{model_name}")


def stream_blob(bucket_name: str, blob_name: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if not blob.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    stream = io.BytesIO()
    blob.download_to_file(stream)
    stream.seek(0)
    return StreamingResponse(
        stream,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={os.path.basename(blob_name)}"},
    )


@app.get("/download_model_file")
def download_model_file(
    model: str,
    artifact: str = "final_model.quant.onnx",
    user_id: str = Depends(current_user),
):
    _validate_name(model, "model")
    if artifact not in _ARTIFACTS:
        raise HTTPException(status_code=400, detail="Unknown artifact")
    return stream_blob(gcs_util.BUCKET_NAME, f"{user_id}/{model}/{artifact}")
