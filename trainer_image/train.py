import os
import shutil

import torch
from ultralytics import YOLO

_DATA_YAML_NAMES = ("data.yaml", "data.yml", "dataset.yaml")


def _find_data_yaml(dataset_dir: str) -> str:
    #the dataset zip can extract to any layout; locate the Ultralytics data.yaml
    for root, _dirs, files in os.walk(dataset_dir):
        for name in files:
            if name in _DATA_YAML_NAMES:
                return os.path.join(root, name)
    raise FileNotFoundError(
        f"No data.yaml found under {dataset_dir}. The dataset zip must contain "
        "an Ultralytics-format data.yaml."
    )


def train(directory: str, dataset: str, epochs: int = 10, batch: int = 16,
          arch: str = "yolov8n"):
    os.makedirs(directory, exist_ok=True)

    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available, training will run on CPU")

    data_yaml = _find_data_yaml(dataset)
    arch = arch[:-3] if arch.endswith(".pt") else arch

    model = YOLO(f"{arch}.pt")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        project=directory,
        name="training_run",
        exist_ok=True,
    )

    best = os.path.join(directory, "training_run", "weights", "best.pt")
    final_pt = os.path.join(directory, "final_model.pt")
    if not os.path.exists(best):
        print(f"Expected trained weights not found at {best}; skipping export")
        return results

    shutil.move(best, final_pt)
    print(f"Final weights: {final_pt}")

    onnx_path = os.path.join(directory, "final_model.onnx")
    try:
        exported = str(YOLO(final_pt).export(format="onnx", imgsz=640, opset=12))
        if os.path.realpath(exported) != os.path.realpath(onnx_path):
            shutil.move(exported, onnx_path)
        print(f"ONNX export: {onnx_path}")
    except Exception as e:
        print(f"ONNX export failed: {e}")
        return results

    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quant_path = os.path.join(directory, "final_model.quant.onnx")
        quantize_dynamic(onnx_path, quant_path, weight_type=QuantType.QInt8)
        print(f"Quantized ONNX: {quant_path}")
    except Exception as e:
        print(f"ONNX quantization skipped: {e}")

    return results
