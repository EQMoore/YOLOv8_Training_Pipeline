import argparse
import os
import sys
import tempfile
import zipfile

import train
import trainer_gcs_util

#artifacts produced by train.train() that we upload back to GCS, if present
_ARTIFACTS = ("final_model.pt", "final_model.onnx", "final_model.quant.onnx")


def _safe_extract(zip_ref: zipfile.ZipFile, dest_dir: str) -> None:
    #guard against zip-slip: every member must resolve inside dest_dir
    dest_root = os.path.realpath(dest_dir)
    for member in zip_ref.namelist():
        target = os.path.realpath(os.path.join(dest_dir, member))
        if target != dest_root and not target.startswith(dest_root + os.sep):
            raise ValueError(f"Unsafe path in zip archive: {member}")
    zip_ref.extractall(dest_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_zip", required=True,
                        help="Full GCS URI of the dataset zip (gs://bucket/path.zip)")
    parser.add_argument("--user_id", required=True,
                        help="Owner id, used as the output path prefix")
    parser.add_argument("--model", required=True,
                        help="Model name, used as the output path prefix")
    parser.add_argument("--arch", default="yolov8n",
                        help="Base YOLO architecture to fine-tune (e.g. yolov8n, yolov8s)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    output_prefix = f"{args.user_id}/{args.model}"

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "dataset.zip")
        trainer_gcs_util.download_from_gcs(zip_path, args.dataset_zip)

        extract_dir = os.path.join(tmp, "dataset")
        os.makedirs(extract_dir, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                _safe_extract(zip_ref, extract_dir)
        except zipfile.BadZipFile:
            print(f"Error: {args.dataset_zip} is not a valid ZIP file", file=sys.stderr)
            sys.exit(1)

        work_dir = os.path.join(tmp, "work")
        train.train(work_dir, extract_dir, epochs=args.epochs, batch=args.batch,
                    arch=args.arch)

        uploaded = []
        for artifact in _ARTIFACTS:
            local_path = os.path.join(work_dir, artifact)
            if os.path.exists(local_path):
                uri = trainer_gcs_util.upload_file(local_path, f"{output_prefix}/{artifact}")
                uploaded.append(uri)
                print(f"Uploaded {uri}")

        if not uploaded:
            print("Training produced no artifacts to upload", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
