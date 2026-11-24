"""Exporta el baseline a ONNX (FP32) y genera una version cuantizada INT8
(cuantizacion estatica post-entrenamiento, formato QDQ, por canal, calibrada con
300 imagenes del subconjunto de entrenamiento).

Salidas: model/model_fp32.onnx, model/model_int8.onnx
"""
import sys
from pathlib import Path

import torch
import torchvision
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_baseline import ROOT, load_fashion, preprocess  # noqa: E402

IMG = 96


class FashionCalib(CalibrationDataReader):
    """300 imagenes de calibracion tomadas SOLO del subconjunto de entrenamiento (las mismas 12.000
    que vio el modelo), para no tocar validacion ni prueba."""

    def __init__(self, n=300, seed=42, train_samples=12000):
        x, _ = load_fashion("train")
        torch.manual_seed(seed)
        tr_idx = torch.randperm(len(x))[:train_samples]  # misma particion que train_baseline.py
        idx = tr_idx[torch.randperm(train_samples, generator=torch.Generator().manual_seed(0))[:n]]
        self.data = iter([{"input": preprocess(x[i:i + 1], IMG).numpy()} for i in idx])

    def get_next(self):
        return next(self.data, None)


def main():
    model = torchvision.models.mobilenet_v2(num_classes=10)
    model.load_state_dict(torch.load(ROOT / "model" / "baseline.pt", map_location="cpu", weights_only=True))
    model.eval()

    fp32 = ROOT / "model" / "model_fp32.onnx"
    pre = ROOT / "model" / "model_fp32_pre.onnx"
    int8 = ROOT / "model" / "model_int8.onnx"
    dummy = torch.randn(1, 3, IMG, IMG)
    torch.onnx.export(model, dummy, fp32, input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    quant_pre_process(str(fp32), str(pre))
    quantize_static(str(pre), str(int8), FashionCalib(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    per_channel=True)
    pre.unlink(missing_ok=True)
    for p in (fp32, int8):
        print(f"{p.name}: {p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
