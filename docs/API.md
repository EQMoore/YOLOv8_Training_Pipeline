# Full reference

Everything the top-level [README](../README.md) quickstart leaves out: the
complete API, architecture, running the trainer directly, GCP setup, tests,
and CI/CD.

## Architecture

```
client ──POST /train_yolo──▶ FastAPI (main.py)
                                │  upload ZIP → gs://$BUCKET_NAME/{user_id}/{model}.zip
                                │  submit Vertex AI CustomContainerTrainingJob
                                ▼
                        Vertex AI runs trainer_image/  (trainer.DOCKERFILE)
                                │  download ZIP, extract, YOLO train, export ONNX, quantize
                                ▼
                        gs://$BUCKET_NAME/{user_id}/{model}/
                            ├── final_model.pt
                            ├── final_model.onnx
                            └── final_model.quant.onnx
```

## Authentication

Every endpoint requires a bearer token:

```
Authorization: Bearer <token>
```

The token is matched (constant-time) against `API_TOKENS` and resolved to a user
id. That user id is the owner prefix for every storage path — it is **never**
taken from the request. Lookups are scoped to the whole `{user_id}/` path
segment, so user `alice` cannot see objects belonging to `alice-corp`. A missing
or unknown token gets `401`.

## API

### `POST /train_yolo` → `202 Accepted`

`multipart/form-data` body:

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `dataset` | file | — | ZIP containing an Ultralytics-format dataset with a `data.yaml` |
| `model` | string | — | Name for this model; used in the output path (`[A-Za-z0-9._-]`, ≤ 64 chars) |
| `arch` | string | `yolov8n` | Base checkpoint to fine-tune (`yolov8n`, `yolov8s`, …) |
| `epochs` | int | `10` | |
| `batch` | int | `16` | |

```bash
curl -X POST "http://localhost:8000/train_yolo" \
  -H "Authorization: Bearer tok_abc" \
  -F "dataset=@data.zip" \
  -F "model=car-detector" \
  -F "epochs=10"
```

The ZIP is uploaded to `gs://$BUCKET_NAME/{user_id}/{model}.zip`, then a Vertex
AI job is submitted.

Response:

```json
{ "job_id": "yolo-train-<user_id>-<uuid>", "user_id": "alice", "model": "car-detector", "status": "submitted" }
```

Errors: `400` (no dataset, or `model` / `arch` fails validation), `401`
(bad token), `409` (`{user_id}/{model}` already has a dataset or artifacts),
`502` (Vertex AI job submission failed — the just-uploaded dataset is deleted so
the name stays free to retry).

Submitting multiple `/train_yolo` calls (different `model` names) trains them
concurrently — nothing in the API serializes job submission; each becomes an
independent Vertex AI job.

### `GET /job_status?job_id=<job_id>`

Returns the state of a submitted job: `{"job_id": ..., "state": "PIPELINE_STATE_RUNNING", "error"?: "..."}`.
`state` is a Vertex AI `PipelineState` (`PIPELINE_STATE_RUNNING`,
`PIPELINE_STATE_SUCCEEDED`, `PIPELINE_STATE_FAILED`, …); `error` is present
only when the job failed. `job_id` embeds its owner, so `404` covers both
"doesn't exist" and "isn't yours" — it never confirms which.

### `GET /get_models`

Returns the list of GCS object names under `{user_id}/` — every dataset and
artifact the caller owns.

### `GET /download_model?model_name=<name>`

Returns the list of GCS object names under `{user_id}/{model_name}/`. This
**lists** a model's files; it does not return their contents — use
`/download_model_file` for that.

### `GET /download_model_file?model=<name>&artifact=<name>`

Streams one artifact as a file download. `artifact` must be one of
`final_model.pt`, `final_model.onnx`, `final_model.quant.onnx` (default:
`final_model.quant.onnx`). `404` if that artifact does not exist.

## Running the API

### Locally

```bash
cd api
pip install -r requirements.txt
export API_TOKENS="tok_dev:me"
uvicorn main:app --reload
```

### As a container

Built from `api/api.DOCKERFILE` with `api/` as the build context:

```bash
docker build api --file api/api.DOCKERFILE --tag yolo-api:local
docker run --rm -p 8000:8080 \
  -e API_TOKENS="tok_dev:me" \
  -e BUCKET_NAME=... -e PROJECT_ID=... -e REGION=us-central1 \
  -e VERTEX_CONTAINER_URI=... \
  yolo-api:local
```

The container listens on `$PORT` (default `8080`) as a non-root user. GCP
credentials come from the environment — Workload Identity on Cloud Run / GKE, or
a mounted key via `GOOGLE_APPLICATION_CREDENTIALS`.

## Trainer container

Built from `../trainer_image/trainer.DOCKERFILE` with `../trainer_image/` as the
build context:

```bash
docker build trainer_image --file trainer_image/trainer.DOCKERFILE --tag yolo-trainer:local
```

CLI (`../trainer_image/main.py`), invoked by Vertex AI:

```
python main.py \
  --dataset_zip=gs://BUCKET/USER/MODEL.zip \
  --user_id=USER --model=MODEL \
  --arch=yolov8n --epochs=10 --batch=16
```

It uploads whichever of `final_model.pt`, `final_model.onnx`,
`final_model.quant.onnx` were produced to `gs://$BUCKET_NAME/{user_id}/{model}/`,
and exits non-zero if the ZIP is invalid or training produces no artifacts.

## GCP setup

One-time provisioning — Artifact Registry, the bucket, service accounts, and
Workload Identity Federation for CI — is scripted in
[`infra/gcp_setup.sh`](INFRA.md). It is idempotent.

## Tests

```bash
cd api
pip install -r requirements-dev.txt
pytest
```

Unit tests (`api/tests/`) cover the API service (`main.py`, `auth.py`,
`gcs_util.py`) with no network or GCP access — the Cloud Storage and Vertex AI
clients are stubbed. `pytest` is configured (in `api/pyproject.toml`) to fail
under **100 % line + branch coverage**. CI runs it with `api/` as the working
directory, which is why the tests and the pytest config live together there.

## CI/CD

| Workflow | Trigger | Does |
| --- | --- | --- |
| **Tests** (`tests.yml`) | PRs to `main`, pushes to other branches | `pytest` (in `api/`) with the 100 % line + branch coverage gate |
| **Docker Image CI** (`build-docker-images.yml`) | PRs to `main`, pushes to other branches | builds `api/api.DOCKERFILE` and `trainer_image/trainer.DOCKERFILE` |
| **Deploy** (`deploy.yml`) | push to `main` | re-runs the tests, then builds + pushes both images to Artifact Registry (`:<sha>` and `:latest`) and deploys the API to Cloud Run |

`main` is a protected branch — the **Tests** check must pass before a PR can be
merged, and **Deploy** re-runs the same suite as a gate before it ships.

### Deploy setup

Run [`infra/gcp_setup.sh`](INFRA.md) once, then set the GitHub Actions
**repository variables** it prints (`GCP_PROJECT_ID`, `GCP_REGION`,
`GCP_BUCKET_NAME`, `GCP_AR_REPO`, `GCP_WIF_PROVIDER`, `GCP_CI_SERVICE_ACCOUNT`,
`GCP_API_SERVICE_ACCOUNT`). Auth is keyless via Workload Identity Federation.
`API_TOKENS` is read from the `api-tokens` Secret Manager secret at runtime.
The Cloud Run service is deployed `--allow-unauthenticated` — the bearer token
is the security boundary; switch the flag in `deploy.yml` for a private service.

## Known limitations

- `GET /job_status` reports pipeline state only, not per-epoch training
  progress — for that, use the Vertex AI console or `gcloud ai custom-jobs
  stream-logs`.
- `download_model_file` buffers the artifact in memory before sending it.
- No request-size limit on the dataset upload.
- No self-service token issuance — `API_TOKENS` is operator-managed only
  (`gcloud secrets versions add`); there is no signup/registration endpoint.
