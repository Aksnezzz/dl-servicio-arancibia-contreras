"""Benchmark de inferencia: PyTorch FP32 vs ONNX Runtime FP32 vs ONNX Runtime INT8.

Cada backend se mide en un proceso aparte (RAM aislada), en el mismo hardware:
  - arranque en frio: carga del modelo + primera prediccion
  - latencia p50/p95 de 1 peticion (preprocesamiento + inferencia), N peticiones
  - throughput secuencial (peticiones/s)
  - RAM (RSS) del proceso tras la carga y durante la inferencia
  - exactitud y F1 macro en el set de prueba completo (10.000 imagenes)

Uso: python scripts/benchmark.py --requests 300
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMG = 96


def run_backend(name: str, n_req: int, threads: int) -> dict:
    import psutil
    from sklearn.metrics import f1_score

    sys.path.insert(0, str(ROOT))
    from app.preprocessing import load_fashion, preprocess

    proc = psutil.Process()
    xte, yte = load_fashion(ROOT, "test")
    rss0 = proc.memory_info().rss

    t0 = time.perf_counter()
    if name == "pytorch_fp32":
        import torch
        import torchvision
        torch.set_num_threads(threads)
        path = ROOT / "model" / "baseline.pt"
        m = torchvision.models.mobilenet_v2(num_classes=10)
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        m.eval()

        def infer(batch):
            with torch.inference_mode():
                return m(torch.from_numpy(batch)).numpy()
    else:
        import onnxruntime as ort
        path = ROOT / "model" / ("model_fp32.onnx" if name == "onnx_fp32" else "model_int8.onnx")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

        def infer(batch):
            return sess.run(None, {"input": batch})[0]
    infer(preprocess(xte[:1], IMG))
    cold = (time.perf_counter() - t0) * 1000
    rss_load = proc.memory_info().rss

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(xte), n_req + 20)
    lat = []
    for k, i in enumerate(idx):
        s = time.perf_counter()
        infer(preprocess(xte[i:i + 1], IMG)).argmax(1)
        if k >= 20:  # 20 peticiones de calentamiento
            lat.append((time.perf_counter() - s) * 1000)
    rss_inf = proc.memory_info().rss
    lat = np.array(lat)

    preds = []
    for i in range(0, len(xte), 250):
        preds.append(infer(preprocess(xte[i:i + 250], IMG)).argmax(1))
    p = np.concatenate(preds)
    y = yte

    return {
        "backend": name,
        "accuracy": round(float((p == y).mean()), 4),
        "f1_macro": round(float(f1_score(y, p, average="macro")), 4),
        "tamano_mb": round(path.stat().st_size / 1e6, 2),
        "arranque_frio_ms": round(cold, 1),
        "p50_ms": round(float(np.percentile(lat, 50)), 2),
        "p95_ms": round(float(np.percentile(lat, 95)), 2),
        "p99_ms": round(float(np.percentile(lat, 99)), 2),
        "media_ms": round(float(lat.mean()), 2),
        "throughput_rps": round(float(1000 / lat.mean()), 1),
        "ram_modelo_mb": round((rss_load - rss0) / 1e6, 1),
        "ram_proceso_mb": round(rss_inf / 1e6, 1),
        "peticiones": len(lat),
        "hilos": threads,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--threads", type=int, default=min(2, os.cpu_count() or 1))
    ap.add_argument("--backends", default="pytorch_fp32,onnx_fp32,onnx_int8")
    ap.add_argument("--worker", default=None)
    ap.add_argument("--out", default=str(ROOT / "reports" / "benchmark.json"))
    a = ap.parse_args()

    if a.worker:
        print(json.dumps(run_backend(a.worker, a.requests, a.threads)))
        return

    results = []
    for b in a.backends.split(","):
        path_needed = ROOT / "model" / {"pytorch_fp32": "baseline.pt", "onnx_fp32": "model_fp32.onnx",
                                         "onnx_int8": "model_int8.onnx"}[b]
        if not path_needed.exists():
            print(f"[omitido] {b}: falta {path_needed.name}")
            continue
        out = subprocess.run([sys.executable, __file__, "--worker", b, "--requests", str(a.requests),
                              "--threads", str(a.threads)], capture_output=True, text=True, check=True)
        r = json.loads(out.stdout.strip().splitlines()[-1])
        results.append(r)
        print(f"{b:13s} acc={r['accuracy']:.4f} p50={r['p50_ms']:.1f}ms p95={r['p95_ms']:.1f}ms "
              f"rps={r['throughput_rps']} RAM={r['ram_proceso_mb']}MB tam={r['tamano_mb']}MB")

    hw = {"cpu": platform.processor() or platform.machine(), "vcpu": os.cpu_count(),
          "so": platform.platform(), "python": platform.python_version()}
    Path(a.out).parent.mkdir(exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"hardware": hw, "resultados": results}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
