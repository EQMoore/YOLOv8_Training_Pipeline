# YOLO Batch Training API

A FastAPI service that turns YOLOv8 fine-tuning into a single authenticated
HTTP call. POST a dataset ZIP and a model name; the service uploads the ZIP to
Cloud Storage and submits a Vertex AI training job that trains with
Ultralytics YOLOv8, exports to ONNX, and quantizes it. Poll for status, then
pull back `final_model.pt` / `.onnx` / `.quant.onnx`. Use it when you want
"train a custom object detector" as an API feature — for an internal tool, a
product, or a pipeline — without hand-rolling GCS upload plumbing, Vertex AI
job submission, or per-tenant storage isolation yourself.

## Prerequisites

- Python 3.12 (for local dev) or Docker
- A GCP project with Vertex AI, Cloud Storage, Artifact Registry, and Secret
  Manager enabled — provision it once with [`infra/gcp_setup.sh`](docs/INFRA.md)
- `gcloud` CLI authenticated to that project
- A dataset ZIP in Ultralytics YOLO format (`images/`, `labels/`, `data.yaml`)

## Quickstart

```bash
git clone <repo-url> && cd YOLOv8_Training_Pipeline/api
pip install -r requirements.txt

export API_TOKENS="tok_dev:me"
export BUCKET_NAME=your-bucket
export PROJECT_ID=your-gcp-project
export REGION=us-central1
export VERTEX_CONTAINER_URI=us-central1-docker.pkg.dev/your-gcp-project/yolo/yolo-trainer:latest

uvicorn main:app --reload
```

In another terminal, submit a training job, check on it, and pull the result:

```bash
JOB=$(curl -s -X POST http://localhost:8000/train_yolo \
  -H "Authorization: Bearer tok_dev" \
  -F "dataset=@dataset.zip" \
  -F "model=car-detector" \
  -F "epochs=20" | python3 -c "import sys,json;print(json.load(sys.stdin)['job_id'])")

curl "http://localhost:8000/job_status?job_id=$JOB" -H "Authorization: Bearer tok_dev"
# {"job_id": "yolo-train-me-...", "state": "PIPELINE_STATE_RUNNING"}

curl "http://localhost:8000/download_model_file?model=car-detector&artifact=final_model.onnx" \
  -H "Authorization: Bearer tok_dev" -o car-detector.onnx
```

## Configuration

The three that matter — without them the service refuses to submit jobs:

| Variable | What it controls |
| --- | --- |
| `API_TOKENS` | `token:username` pairs, comma-separated. This is the entire auth system — no signup flow, tokens are operator-issued. |
| `BUCKET_NAME`, `PROJECT_ID`, `REGION` | Where datasets/artifacts live and which Vertex AI region runs jobs. All three are required together. |
| `VERTEX_CONTAINER_URI` | The trainer image Vertex runs per job — build it from `trainer_image/trainer.DOCKERFILE` and push it to Artifact Registry first. |

Everything else (GPU vs. CPU training, machine type, port) is documented in
the full reference below.

## Project structure

```
api/                the API service — FastAPI app (main.py), bearer auth (auth.py), GCS/Vertex AI glue (gcs_util.py)
api/tests/           unit tests for the API service (100% coverage gate in CI)
trainer_image/       the trainer — its own image, run by Vertex AI, not by the API process
infra/               one-time GCP provisioning (gcp_setup.sh)
docs/                full API reference (API.md) and GCP setup detail (INFRA.md)
.github/workflows/    CI (tests, Docker builds) and CD (build + deploy to Cloud Run on push to main)
```

## Full docs

API reference (all endpoints, error codes), deployment/CI-CD, and known
limitations: [docs/API.md](docs/API.md)
