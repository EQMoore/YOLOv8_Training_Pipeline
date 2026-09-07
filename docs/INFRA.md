# GCP setup

`../infra/gcp_setup.sh` provisions everything the API and its CI need. Run it once, from
a machine authenticated with an account that has Owner (or an equivalent set of
admin roles) on the project.

```bash
PROJECT_ID=my-project \
BUCKET_NAME=my-yolo-bucket \
GITHUB_REPO=my-org/Yolov8_Training_Pipeline \
REGION=us-central1 \
  ./infra/gcp_setup.sh
```

It is idempotent — re-running it is safe.

## What it creates

| Resource | Purpose |
| --- | --- |
| Enabled APIs | Artifact Registry, Vertex AI, Cloud Storage, Cloud Run, IAM Credentials, STS, Secret Manager |
| Artifact Registry repo `yolo` | Holds the trainer and API images |
| GCS bucket | Datasets, training artifacts, Vertex staging |
| Service account `yolo-api` | Cloud Run runtime identity — `aiplatform.user` + `storage.objectAdmin` on the bucket |
| Vertex custom-code service agent | Granted `storage.objectAdmin` on the bucket (it runs the training container) |
| Service account `yolo-ci` | GitHub Actions identity — `artifactregistry.writer`, `run.admin`, and `serviceAccountUser` on `yolo-api` |
| Secret `api-tokens` | Holds `API_TOKENS`; created with a placeholder, readable by `yolo-api` (`secretmanager.secretAccessor`) |
| Workload Identity pool/provider `github` | Lets GitHub Actions impersonate `yolo-ci` with no key file; locked to `GITHUB_REPO` |

## After running it

1. The script prints a block of values. Add them as **GitHub Actions repository
   variables** (`Settings → Secrets and variables → Actions → Variables`):
   `GCP_PROJECT_ID`, `GCP_REGION`, `GCP_BUCKET_NAME`, `GCP_AR_REPO`,
   `GCP_WIF_PROVIDER`, `GCP_CI_SERVICE_ACCOUNT`, `GCP_API_SERVICE_ACCOUNT`.
2. Replace the placeholder API token with real values:
   ```bash
   printf 'tok_xxx:alice' | gcloud secrets versions add api-tokens --data-file=-
   ```
3. Push to `main` — the **Deploy** workflow builds both images, pushes them to
   Artifact Registry, and deploys the API to Cloud Run. It wires
   `VERTEX_CONTAINER_URI` to the trainer image built in the same run.

## Teardown

```bash
gcloud run services delete yolo-api --region=REGION
gcloud artifacts repositories delete yolo --location=REGION
gcloud storage rm -r gs://BUCKET_NAME
gcloud secrets delete api-tokens
gcloud iam service-accounts delete yolo-api@PROJECT_ID.iam.gserviceaccount.com
gcloud iam service-accounts delete yolo-ci@PROJECT_ID.iam.gserviceaccount.com
gcloud iam workload-identity-pools delete github --location=global
```
