#!/usr/bin/env bash
#Credit : Claude

# One-time GCP setup for the YOLO Batch Training API.
#
# Creates: enabled APIs, an Artifact Registry Docker repo, the GCS bucket,
# a runtime service account for the API, a CI service account for GitHub
# Actions, and a Workload Identity Federation pool/provider so GitHub Actions
# can authenticate without a downloaded key.
#
# Safe to re-run: every step is idempotent.
#
# Usage:
#   PROJECT_ID=my-proj BUCKET_NAME=my-bucket GITHUB_REPO=owner/repo \
#     ./infra/gcp_setup.sh
#
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
BUCKET_NAME="${BUCKET_NAME:?set BUCKET_NAME (bucket name only, no gs://)}"
GITHUB_REPO="${GITHUB_REPO:?set GITHUB_REPO as owner/repo}"
REGION="${REGION:-us-central1}"
AR_REPO="${AR_REPO:-yolo}"

API_SA="yolo-api"
CI_SA="yolo-ci"
POOL="github"
PROVIDER="github"

API_SA_EMAIL="${API_SA}@${PROJECT_ID}.iam.gserviceaccount.com"
CI_SA_EMAIL="${CI_SA}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud config set project "$PROJECT_ID"
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"

# --- helpers -----------------------------------------------------------------

# Create a service account if it does not exist, then block until it is
# visible to IAM. Service-account creation is eventually consistent, and a
# policy binding that references one too soon fails with "does not exist".
ensure_sa() {
  local name="$1" display="$2"
  local email="${name}@${PROJECT_ID}.iam.gserviceaccount.com"
  if ! gcloud iam service-accounts describe "$email" >/dev/null 2>&1; then
    gcloud iam service-accounts create "$name" --display-name="$display"
  fi
  local i
  for i in $(seq 1 30); do
    if gcloud iam service-accounts describe "$email" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "ERROR: ${email} still not visible after 60s" >&2
  return 1
}

# Retry a command with linear backoff. IAM policy bindings can return a
# transient INVALID_ARGUMENT for a few seconds after their member (a service
# account or a WIF principal) is created.
retry() {
  local n=0
  until "$@"; do
    n=$((n + 1))
    if [ "$n" -ge 6 ]; then
      echo "ERROR: gave up after ${n} attempts: $*" >&2
      return 1
    fi
    echo "  retry ${n}/5 in $((n * 5))s ..." >&2
    sleep $((n * 5))
  done
}

echo "== Enabling APIs =="
gcloud services enable \
  artifactregistry.googleapis.com \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  run.googleapis.com \
  iamcredentials.googleapis.com \
  sts.googleapis.com \
  secretmanager.googleapis.com

echo "== Artifact Registry repo =="
gcloud artifacts repositories create "$AR_REPO" \
  --repository-format=docker --location="$REGION" \
  --description="YOLO trainer / API images" \
  || echo "  repo already exists"

echo "== GCS bucket =="
gcloud storage buckets create "gs://${BUCKET_NAME}" \
  --location="$REGION" --uniform-bucket-level-access \
  || echo "  bucket already exists"

echo "== API runtime service account =="
ensure_sa "$API_SA" "YOLO API (Cloud Run runtime)"
# submit Vertex AI training jobs
retry gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${API_SA_EMAIL}" \
  --role="roles/aiplatform.user" --condition=None
# read datasets / write artifacts / write the Vertex staging prefix
retry gcloud storage buckets add-iam-policy-binding "gs://${BUCKET_NAME}" \
  --member="serviceAccount:${API_SA_EMAIL}" \
  --role="roles/storage.objectAdmin"

echo "== Vertex custom-training service agent =="
# CustomContainerTrainingJob runs as this agent by default; it needs the bucket
CC_AGENT="service-${PROJECT_NUMBER}@gcp-sa-aiplatform-cc.iam.gserviceaccount.com"
gcloud beta services identity create --service=aiplatform.googleapis.com --project="$PROJECT_ID" >/dev/null 2>&1 || true
retry gcloud storage buckets add-iam-policy-binding "gs://${BUCKET_NAME}" \
  --member="serviceAccount:${CC_AGENT}" \
  --role="roles/storage.objectAdmin"

echo "== CI service account (GitHub Actions) =="
ensure_sa "$CI_SA" "YOLO CI (GitHub Actions)"
# push images
retry gcloud artifacts repositories add-iam-policy-binding "$AR_REPO" \
  --location="$REGION" \
  --member="serviceAccount:${CI_SA_EMAIL}" \
  --role="roles/artifactregistry.writer"
# deploy Cloud Run, and act as the API runtime SA while doing so
retry gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${CI_SA_EMAIL}" \
  --role="roles/run.admin" --condition=None
retry gcloud iam service-accounts add-iam-policy-binding "$API_SA_EMAIL" \
  --member="serviceAccount:${CI_SA_EMAIL}" \
  --role="roles/iam.serviceAccountUser"

echo "== API token secret =="
if ! gcloud secrets describe api-tokens >/dev/null 2>&1; then
  printf 'changeme:demo' | gcloud secrets create api-tokens --data-file=-
  echo "  created 'api-tokens' with a PLACEHOLDER value — add a real version:"
  echo "    printf 'tok_xxx:alice' | gcloud secrets versions add api-tokens --data-file=-"
fi
# the Cloud Run service (running as the API SA) reads this at startup
retry gcloud secrets add-iam-policy-binding api-tokens \
  --member="serviceAccount:${API_SA_EMAIL}" \
  --role="roles/secretmanager.secretAccessor"

echo "== Workload Identity Federation =="
gcloud iam workload-identity-pools create "$POOL" \
  --location=global --display-name="GitHub Actions" \
  || echo "  pool already exists"
gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" \
  --location=global --workload-identity-pool="$POOL" \
  --display-name="GitHub" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='${GITHUB_REPO}'" \
  || echo "  provider already exists"
# only this repo may impersonate the CI service account
retry gcloud iam service-accounts add-iam-policy-binding "$CI_SA_EMAIL" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/attribute.repository/${GITHUB_REPO}"

WIF_PROVIDER="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/providers/${PROVIDER}"

cat <<EOF

== Done ==

Set these as GitHub Actions repository variables (Settings > Secrets and
variables > Actions > Variables):

  GCP_PROJECT_ID          ${PROJECT_ID}
  GCP_REGION              ${REGION}
  GCP_BUCKET_NAME         ${BUCKET_NAME}
  GCP_AR_REPO             ${AR_REPO}
  GCP_WIF_PROVIDER        ${WIF_PROVIDER}
  GCP_CI_SERVICE_ACCOUNT  ${CI_SA_EMAIL}
  GCP_API_SERVICE_ACCOUNT ${API_SA_EMAIL}

The Deploy workflow builds and pushes both images and deploys the API to Cloud
Run on every push to main. It sets VERTEX_CONTAINER_URI to the trainer image it
just built:

  ${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/yolo-trainer:<commit-sha>

Put the real API tokens into the 'api-tokens' secret this script created:

  printf 'tok_xxx:alice' | gcloud secrets versions add api-tokens --data-file=-
EOF
