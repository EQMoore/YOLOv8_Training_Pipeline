import pytest

import gcs_util


class _FakeBlob:
    def __init__(self, name):
        self.name = name
        self.uploaded_from = None
        self.deleted = False

    def upload_from_filename(self, local_path):
        self.uploaded_from = local_path

    def delete(self):
        self.deleted = True


class _FakeBucket:
    def __init__(self, existing=()):
        self._existing = [_FakeBlob(n) for n in existing]
        self.created = {}

    def list_blobs(self, prefix=None):
        prefix = prefix or ""
        return [b for b in self._existing if b.name.startswith(prefix)]

    def blob(self, path):
        blob = _FakeBlob(path)
        self.created[path] = blob
        return blob


class _FakeStorageClient:
    def __init__(self, bucket):
        self._bucket = bucket

    def bucket(self, name):
        assert name == gcs_util.BUCKET_NAME
        return self._bucket


def _use(monkeypatch, bucket):
    monkeypatch.setattr(
        gcs_util.storage, "Client", lambda *a, **k: _FakeStorageClient(bucket)
    )


@pytest.fixture
def bucket(monkeypatch):
    b = _FakeBucket()
    _use(monkeypatch, b)
    return b


def test_get_user_models_returns_matching_names(monkeypatch):
    b = _FakeBucket(existing=["alice/m.zip", "alice/m/final_model.pt", "bob/x.zip"])
    _use(monkeypatch, b)
    assert gcs_util.get_user_models("alice") == [
        "alice/m.zip",
        "alice/m/final_model.pt",
    ]


def test_get_user_models_is_segment_scoped(monkeypatch):
    #"alice" must not match the separate tenant "alice-corp"
    b = _FakeBucket(existing=["alice/m.zip", "alice-corp/secret.zip"])
    _use(monkeypatch, b)
    assert gcs_util.get_user_models("alice") == ["alice/m.zip"]


def test_get_user_models_scopes_to_one_model(monkeypatch):
    b = _FakeBucket(existing=["alice/car/final_model.pt", "alice/car-v2/final_model.pt"])
    _use(monkeypatch, b)
    assert gcs_util.get_user_models("alice/car") == ["alice/car/final_model.pt"]


def test_get_user_models_empty(monkeypatch):
    _use(monkeypatch, _FakeBucket())
    assert gcs_util.get_user_models("nobody") == []


def test_check_gcs_unique_name_matches_dataset_zip(monkeypatch):
    _use(monkeypatch, _FakeBucket(existing=["alice/m.zip"]))
    assert gcs_util.check_gcs_unique_name("alice/m") is True


def test_check_gcs_unique_name_matches_artifacts(monkeypatch):
    _use(monkeypatch, _FakeBucket(existing=["alice/m/final_model.pt"]))
    assert gcs_util.check_gcs_unique_name("alice/m") is True


def test_check_gcs_unique_name_ignores_sibling_prefix(monkeypatch):
    #checking "alice/model" must not be tripped by "alice/model-x" / "alice/models"
    b = _FakeBucket(existing=["alice/model-x.zip", "alice/models/y.pt"])
    _use(monkeypatch, b)
    assert gcs_util.check_gcs_unique_name("alice/model") is False


def test_check_gcs_unique_name_false_when_absent(monkeypatch):
    _use(monkeypatch, _FakeBucket())
    assert gcs_util.check_gcs_unique_name("alice/m") is False


def test_upload_to_gcs(bucket):
    uri = gcs_util.upload_to_gcs("/tmp/data.zip", "alice/m.zip")
    assert uri == f"gs://{gcs_util.BUCKET_NAME}/alice/m.zip"
    assert bucket.created["alice/m.zip"].uploaded_from == "/tmp/data.zip"


def test_delete_from_gcs(bucket):
    gcs_util.delete_from_gcs("alice/m.zip")
    assert bucket.created["alice/m.zip"].deleted is True


class _FakeTrainingJob:
    instances = []
    list_results = []
    last_list_filter = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.submitted = None
        _FakeTrainingJob.instances.append(self)

    def submit(self, **kwargs):
        self.submitted = kwargs

    @classmethod
    def list(cls, filter=None):
        cls.last_list_filter = filter
        return cls.list_results


class _FakeAiplatform:
    CustomContainerTrainingJob = _FakeTrainingJob

    def __init__(self):
        self.init_kwargs = None

    def init(self, **kwargs):
        self.init_kwargs = kwargs


@pytest.fixture
def vertex(monkeypatch):
    _FakeTrainingJob.instances = []
    _FakeTrainingJob.list_results = []
    _FakeTrainingJob.last_list_filter = None
    fake = _FakeAiplatform()
    monkeypatch.setattr(gcs_util, "aiplatform", fake)
    monkeypatch.setattr(gcs_util, "BUCKET_NAME", "bkt")
    monkeypatch.setattr(gcs_util, "PROJECT_ID", "proj")
    monkeypatch.setattr(gcs_util, "REGION", "us-central1")
    monkeypatch.setattr(gcs_util, "VERTEX_CONTAINER_URI", "img:latest")
    for var in ("TRAIN_ACCELERATOR_TYPE", "TRAIN_ACCELERATOR_COUNT", "TRAIN_MACHINE_TYPE"):
        monkeypatch.delenv(var, raising=False)
    return fake


def test_submit_training_job_success(vertex):
    job_id = gcs_util.submit_training_job(
        "gs://bkt/alice/m.zip", "alice", "m", 5, 8, "yolov8s"
    )
    assert job_id.startswith("yolo-train-alice-")
    assert vertex.init_kwargs == {
        "project": "proj",
        "location": "us-central1",
        "staging_bucket": "gs://bkt",
    }
    job = _FakeTrainingJob.instances[-1]
    assert job.kwargs["container_uri"] == "img:latest"
    assert job.submitted["args"] == [
        "--dataset_zip=gs://bkt/alice/m.zip",
        "--user_id=alice",
        "--model=m",
        "--arch=yolov8s",
        "--epochs=5",
        "--batch=8",
    ]
    assert job.submitted["base_output_dir"] == f"gs://bkt/training_outputs/{job_id}"
    #the trainer container needs BUCKET_NAME itself to upload artifacts
    assert job.submitted["environment_variables"] == {"BUCKET_NAME": "bkt"}
    #no accelerator env -> CPU-only job
    assert job.submitted["machine_type"] == "n1-standard-8"
    assert "accelerator_type" not in job.submitted
    assert "accelerator_count" not in job.submitted


def test_submit_training_job_uses_default_arch(vertex):
    gcs_util.submit_training_job("gs://bkt/a/m.zip", "a", "m", 10, 16)
    assert "--arch=yolov8n" in _FakeTrainingJob.instances[-1].submitted["args"]


def test_submit_training_job_attaches_gpu_when_configured(vertex, monkeypatch):
    monkeypatch.setenv("TRAIN_ACCELERATOR_TYPE", "NVIDIA_TESLA_T4")
    monkeypatch.setenv("TRAIN_ACCELERATOR_COUNT", "2")
    monkeypatch.setenv("TRAIN_MACHINE_TYPE", "n1-standard-16")
    gcs_util.submit_training_job("gs://bkt/a/m.zip", "a", "m", 10, 16)
    job = _FakeTrainingJob.instances[-1]
    assert job.submitted["machine_type"] == "n1-standard-16"
    assert job.submitted["accelerator_type"] == "NVIDIA_TESLA_T4"
    assert job.submitted["accelerator_count"] == 2


class _FakeState:
    def __init__(self, name):
        self.name = name


class _FakeGcaResource:
    def __init__(self, error_message=""):
        class _Error:
            def __init__(self, message):
                self.message = message

        self.error = _Error(error_message)


class _FakeListedJob:
    #a job as returned by CustomContainerTrainingJob.list()
    def __init__(self, state_name, error_message=""):
        self.state = _FakeState(state_name)
        self._gca_resource = _FakeGcaResource(error_message)


class _FakeListedJobNoDetail:
    #mimics an SDK version where the private error-detail attribute is absent
    def __init__(self, state_name):
        self.state = _FakeState(state_name)


def test_get_training_status_rejects_mismatched_owner(vertex):
    #job_id doesn't embed this caller's user_id -> refused without querying Vertex
    assert gcs_util.get_training_status("yolo-train-bob-abc123", "alice") is None
    assert _FakeTrainingJob.last_list_filter is None


def test_get_training_status_not_found(vertex):
    result = gcs_util.get_training_status("yolo-train-alice-abc123", "alice")
    assert result is None
    assert _FakeTrainingJob.last_list_filter == 'display_name="yolo-train-alice-abc123"'


def test_get_training_status_running(vertex):
    _FakeTrainingJob.list_results = [_FakeListedJob("PIPELINE_STATE_RUNNING")]
    result = gcs_util.get_training_status("yolo-train-alice-abc123", "alice")
    assert result == {"job_id": "yolo-train-alice-abc123", "state": "PIPELINE_STATE_RUNNING"}


def test_get_training_status_failed_includes_error(vertex):
    _FakeTrainingJob.list_results = [
        _FakeListedJob("PIPELINE_STATE_FAILED", "quota exceeded")
    ]
    result = gcs_util.get_training_status("yolo-train-alice-abc123", "alice")
    assert result == {
        "job_id": "yolo-train-alice-abc123",
        "state": "PIPELINE_STATE_FAILED",
        "error": "quota exceeded",
    }


def test_get_training_status_tolerates_missing_error_detail(vertex):
    _FakeTrainingJob.list_results = [_FakeListedJobNoDetail("PIPELINE_STATE_RUNNING")]
    result = gcs_util.get_training_status("yolo-train-alice-abc123", "alice")
    assert result == {"job_id": "yolo-train-alice-abc123", "state": "PIPELINE_STATE_RUNNING"}


def test_get_training_status_missing_env(monkeypatch):
    monkeypatch.setattr(gcs_util, "PROJECT_ID", None)
    with pytest.raises(RuntimeError):
        gcs_util.get_training_status("yolo-train-alice-abc123", "alice")


def test_submit_training_job_missing_container_uri(monkeypatch):
    monkeypatch.setattr(gcs_util, "VERTEX_CONTAINER_URI", "default")
    with pytest.raises(RuntimeError):
        gcs_util.submit_training_job("gs://b/x.zip", "a", "m", 10, 16)


def test_submit_training_job_missing_bucket(monkeypatch):
    monkeypatch.setattr(gcs_util, "BUCKET_NAME", None)
    monkeypatch.setattr(gcs_util, "VERTEX_CONTAINER_URI", "img:latest")
    with pytest.raises(RuntimeError):
        gcs_util.submit_training_job("gs://b/x.zip", "a", "m", 10, 16)
