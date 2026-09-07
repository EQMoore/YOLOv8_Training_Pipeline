import io

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

import main
from auth import current_user


@pytest.fixture
def client():
    main.app.dependency_overrides[current_user] = lambda: "alice"
    with TestClient(main.app) as c:
        yield c
    main.app.dependency_overrides.clear()


@pytest.fixture
def stub_gcs(monkeypatch):
    monkeypatch.setattr(main.gcs_util, "check_gcs_unique_name", lambda name: False)
    monkeypatch.setattr(
        main.gcs_util, "upload_to_gcs", lambda path, blob: f"gs://bucket/{blob}"
    )
    monkeypatch.setattr(main.gcs_util, "submit_training_job", lambda *args: "yolo-train-123")
    monkeypatch.setattr(main.gcs_util, "delete_from_gcs", lambda blob: None)


def _dataset(content=b"PK\x03\x04data"):
    return {"dataset": ("data.zip", content, "application/zip")}


def test_train_yolo_success(client, stub_gcs):
    resp = client.post(
        "/train_yolo",
        data={"model": "car-detector", "epochs": "5", "batch": "8", "arch": "yolov8n"},
        files=_dataset(),
    )
    assert resp.status_code == 202
    assert resp.json() == {
        "job_id": "yolo-train-123",
        "user_id": "alice",
        "model": "car-detector",
        "status": "submitted",
    }


def test_train_yolo_applies_form_defaults(client, stub_gcs):
    resp = client.post("/train_yolo", data={"model": "m"}, files=_dataset())
    assert resp.status_code == 202


def test_train_yolo_rejects_missing_filename():
    with pytest.raises(HTTPException) as exc:
        main.train_yolo(
            dataset=UploadFile(filename=None, file=io.BytesIO(b"x")),
            model="m",
            epochs=10,
            batch=16,
            arch="yolov8n",
            user_id="alice",
        )
    assert exc.value.status_code == 400


def test_train_yolo_rejects_invalid_model(client, stub_gcs):
    resp = client.post("/train_yolo", data={"model": "bad name"}, files=_dataset())
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Invalid model"


def test_train_yolo_rejects_invalid_arch(client, stub_gcs):
    resp = client.post(
        "/train_yolo", data={"model": "ok", "arch": "bad/arch"}, files=_dataset()
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Invalid arch"


def test_train_yolo_rejects_duplicate_name(client, monkeypatch):
    monkeypatch.setattr(main.gcs_util, "check_gcs_unique_name", lambda name: True)
    resp = client.post("/train_yolo", data={"model": "dupe"}, files=_dataset())
    assert resp.status_code == 409


def test_train_yolo_ignores_temp_cleanup_failure(client, stub_gcs, monkeypatch):
    def boom(path):
        raise OSError("cannot remove")

    monkeypatch.setattr(main.os, "remove", boom)
    resp = client.post("/train_yolo", data={"model": "m"}, files=_dataset())
    assert resp.status_code == 202


def test_train_yolo_removes_dataset_when_submit_fails(client, monkeypatch):
    deleted = []

    def fail_submit(*args):
        raise RuntimeError("vertex unavailable")

    monkeypatch.setattr(main.gcs_util, "check_gcs_unique_name", lambda name: False)
    monkeypatch.setattr(main.gcs_util, "upload_to_gcs", lambda path, blob: f"gs://b/{blob}")
    monkeypatch.setattr(main.gcs_util, "submit_training_job", fail_submit)
    monkeypatch.setattr(main.gcs_util, "delete_from_gcs", lambda blob: deleted.append(blob))

    resp = client.post("/train_yolo", data={"model": "m"}, files=_dataset())
    assert resp.status_code == 502
    assert deleted == ["alice/m.zip"]


def test_train_yolo_submit_failure_survives_cleanup_error(client, monkeypatch):
    def fail_submit(*args):
        raise RuntimeError("vertex unavailable")

    def fail_delete(blob):
        raise RuntimeError("gcs delete failed")

    monkeypatch.setattr(main.gcs_util, "check_gcs_unique_name", lambda name: False)
    monkeypatch.setattr(main.gcs_util, "upload_to_gcs", lambda path, blob: "gs://b/x")
    monkeypatch.setattr(main.gcs_util, "submit_training_job", fail_submit)
    monkeypatch.setattr(main.gcs_util, "delete_from_gcs", fail_delete)

    resp = client.post("/train_yolo", data={"model": "m"}, files=_dataset())
    assert resp.status_code == 502


def test_job_status_rejects_invalid_id(client):
    resp = client.get("/job_status", params={"job_id": "bad id!"})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Invalid job_id"


def test_job_status_not_found(client, monkeypatch):
    monkeypatch.setattr(main.gcs_util, "get_training_status", lambda job_id, user_id: None)
    resp = client.get("/job_status", params={"job_id": "yolo-train-alice-abc123"})
    assert resp.status_code == 404


def test_job_status_returns_state(client, monkeypatch):
    seen = {}

    def fake_status(job_id, user_id):
        seen["job_id"] = job_id
        seen["user_id"] = user_id
        return {"job_id": job_id, "state": "PIPELINE_STATE_RUNNING"}

    monkeypatch.setattr(main.gcs_util, "get_training_status", fake_status)
    resp = client.get("/job_status", params={"job_id": "yolo-train-alice-abc123"})
    assert resp.status_code == 200
    assert resp.json() == {"job_id": "yolo-train-alice-abc123", "state": "PIPELINE_STATE_RUNNING"}
    assert seen == {"job_id": "yolo-train-alice-abc123", "user_id": "alice"}


def test_get_models(client, monkeypatch):
    monkeypatch.setattr(main.gcs_util, "get_user_models", lambda uid: [f"{uid}/m.zip"])
    resp = client.get("/get_models")
    assert resp.status_code == 200
    assert resp.json() == ["alice/m.zip"]


def test_download_model_lists_objects(client, monkeypatch):
    monkeypatch.setattr(main.gcs_util, "get_user_models", lambda path: [path])
    resp = client.get("/download_model", params={"model_name": "car-detector"})
    assert resp.status_code == 200
    assert resp.json() == ["alice/car-detector"]


def test_download_model_rejects_invalid_name(client):
    resp = client.get("/download_model", params={"model_name": "bad name"})
    assert resp.status_code == 400


class _FakeBlob:
    def __init__(self, exists):
        self._exists = exists

    def exists(self):
        return self._exists

    def download_to_file(self, stream):
        stream.write(b"model-bytes")


class _FakeStorageClient:
    def __init__(self, blob):
        self._blob = blob

    def bucket(self, name):
        return self

    def blob(self, name):
        return self._blob


@pytest.fixture
def fake_storage(monkeypatch):
    def factory(exists=True):
        blob = _FakeBlob(exists)
        monkeypatch.setattr(
            main.storage, "Client", lambda *a, **k: _FakeStorageClient(blob)
        )
        return blob

    return factory


def test_download_model_file_streams_artifact(client, fake_storage):
    fake_storage(exists=True)
    resp = client.get("/download_model_file", params={"model": "car-detector"})
    assert resp.status_code == 200
    assert resp.content == b"model-bytes"
    assert "attachment" in resp.headers["content-disposition"]


def test_download_model_file_missing_artifact_returns_404(client, fake_storage):
    fake_storage(exists=False)
    resp = client.get("/download_model_file", params={"model": "car-detector"})
    assert resp.status_code == 404


def test_download_model_file_rejects_unknown_artifact(client):
    resp = client.get(
        "/download_model_file", params={"model": "m", "artifact": "secrets.txt"}
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Unknown artifact"


def test_download_model_file_rejects_invalid_model(client):
    resp = client.get("/download_model_file", params={"model": "bad name"})
    assert resp.status_code == 400
