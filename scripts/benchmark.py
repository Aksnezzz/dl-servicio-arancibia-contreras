"""Benchmark de inferencia: PyTorch FP32 vs ONNX Runtime FP32 vs ONNX Runtime INT8.

Cada backend se mide en un proceso aparte, en el mismo hardware. Se separan los tiempos:
  A) Flujo real de /predict con fotos JPEG de ~1,8 MB (3000x3000):
     decodificacion -> preprocesamiento (gris, recorte, 28x28, 96x96x3) -> inferencia
  B) Solo inferencia (tensor 1x3x96x96 ya preparado)
y se reporta p50/p95/p99 de >= 100 peticiones de 1 imagen (20 de calentamiento),
throughput secuencial, RAM (RSS del proceso y uso maximo del contenedor si hay cgroup),
arranque en frio (importar + cargar modelo + 1a prediccion) y exactitud en prueba (10.000).

Uso local:      python scripts/benchmark.py --requests 300
Con limites:    ver README (docker run --memory=4g --cpus=2 ...)
"""
import argparse
import io
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMG = 96
FILES = {"pytorch_fp32": "baseline.pt", "onnx_fp32": "model_fp32.onnx", "onnx_int8": "model_int8.onnx"}


def pct(a, q):
    import numpy as np
    return round(float(np.percentile(a, q)), 2)


def cgroup_peak_mb():
    for p in ("/sys/fs/cgroup/memory/memory.max_usage_in_bytes", "/sys/fs/cgroup/memory.peak"):
        if os.path.exists(p):
            return round(int(open(p).read()) / 1e6, 1)
    return None


def run_backend(name: str, n_req: int, threads: int, t_start: float) -> dict:
    import numpy as np
    import psutil
    from PIL import Image
    sys.path.insert(0, str(ROOT))
    from app.preprocessing import image_bytes_to_array, load_fashion, preprocess

    proc = psutil.Process()
    path = ROOT / "model" / FILES[name]
    if name == "pytorch_fp32":
        import torch
        import torchvision
        torch.set_num_threads(threads)
        m = torchvision.models.mobilenet_v2(num_classes=10)
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        m.eval()

        def infer(batch):
            with torch.inference_mode():
                return m(torch.from_numpy(batch)).numpy()
    else:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

        def infer(batch):
            return sess.run(None, {"input": batch})[0]

    jpegs = [p.read_bytes() for p in sorted((ROOT / "data" / "jpeg_sinteticas").glob("*.jpg"))]
    if not jpegs:
        raise SystemExit("faltan fotos: python scripts/make_test_jpegs.py")
    first = preprocess(image_bytes_to_array(Image.open(io.BytesIO(jpegs[0])))[None], IMG)
    infer(first)
    cold = (time.perf_counter() - t_start) * 1000  # desde el inicio del proceso
    rss_load = proc.memory_info().rss

    # A) flujo real con JPEG
    dec, pre, inf, tot = [], [], [], []
    for k in range(n_req + 20):
        b = jpegs[k % len(jpegs)]
        t0 = time.perf_counter()
        arr = image_bytes_to_array(Image.open(io.BytesIO(b)))
        t1 = time.perf_counter()
        x = preprocess(arr[None], IMG)
        t2 = time.perf_counter()
        infer(x).argmax(1)
        t3 = time.perf_counter()
        if k >= 20:
            dec.append((t1 - t0) * 1e3)
            pre.append((t2 - t1) * 1e3)
            inf.append((t3 - t2) * 1e3)
            tot.append((t3 - t0) * 1e3)

    # B) solo inferencia, tensor ya preparado
    xte, yte = load_fashion(ROOT, "test")
    rng = np.random.default_rng(0)
    ready = [preprocess(xte[i:i + 1], IMG) for i in rng.integers(0, len(xte), 50)]
    only = []
    for k in range(n_req + 20):
        t0 = time.perf_counter()
        infer(ready[k % len(ready)])
        if k >= 20:
            only.append((time.perf_counter() - t0) * 1e3)

    # RAM medida tras atender las peticiones de 1 imagen (como el servicio), antes de evaluar exactitud
    rss_serv = proc.memory_info().rss
    peak = cgroup_peak_mb()

    from sklearn.metrics import f1_score
    preds = [infer(preprocess(xte[i:i + 25], IMG)).argmax(1) for i in range(0, len(xte), 25)]
    p = np.concatenate(preds)
    tot = np.array(tot)
    return {
        "backend": name,
        "accuracy_test": round(float((p == yte).mean()), 4),
        "f1_macro_test": round(float(f1_score(yte, p, average="macro")), 4),
        "tamano_mb": round(path.stat().st_size / 1e6, 2),
        "jpeg_total_p50_ms": pct(tot, 50), "jpeg_total_p95_ms": pct(tot, 95), "jpeg_total_p99_ms": pct(tot, 99),
        "jpeg_decodificacion_p50_ms": pct(dec, 50), "jpeg_decodificacion_p95_ms": pct(dec, 95),
        "preprocesamiento_p50_ms": pct(pre, 50), "preprocesamiento_p95_ms": pct(pre, 95),
        "inferencia_en_flujo_p50_ms": pct(inf, 50), "inferencia_en_flujo_p95_ms": pct(inf, 95),
        "solo_inferencia_p50_ms": pct(only, 50), "solo_inferencia_p95_ms": pct(only, 95),
        "throughput_jpeg_rps": round(float(1000 / tot.mean()), 1),
        "throughput_solo_inferencia_rps": round(float(1000 / np.mean(only)), 1),
        "ram_proceso_tras_carga_mb": round(rss_load / 1e6, 1),
        "ram_proceso_mb": round(rss_serv / 1e6, 1),
        "ram_contenedor_pico_mb": peak,
        "arranque_frio_ms": round(cold, 1),
        "peticiones": len(tot),
        "hilos": threads,
    }


def main():
    t_start = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--threads", type=int, default=min(2, os.cpu_count() or 1))
    ap.add_argument("--backends", default="pytorch_fp32,onnx_fp32,onnx_int8")
    ap.add_argument("--worker", default=None)
    ap.add_argument("--hardware-note", default="")
    ap.add_argument("--out", default=str(ROOT / "reports" / "benchmark.json"))
    ap.add_argument("--append", action="store_true", help="agrega resultados a --out en vez de reemplazar")
    a = ap.parse_args()

    if a.worker:
        print(json.dumps(run_backend(a.worker, a.requests, a.threads, t_start)))
        return

    results = []
    for b in a.backends.split(","):
        if not (ROOT / "model" / FILES[b]).exists():
            print(f"[omitido] {b}: falta model/{FILES[b]}")
            continue
        out = subprocess.run([sys.executable, __file__, "--worker", b, "--requests", str(a.requests),
                              "--threads", str(a.threads)], capture_output=True, text=True, check=True)
        r = json.loads(out.stdout.strip().splitlines()[-1])
        results.append(r)
        print(f"{b:13s} acc={r['accuracy_test']:.4f} jpeg p50/p95={r['jpeg_total_p50_ms']}/{r['jpeg_total_p95_ms']} ms "
              f"(dec {r['jpeg_decodificacion_p50_ms']} + pre {r['preprocesamiento_p50_ms']} + "
              f"inf {r['inferencia_en_flujo_p50_ms']}) | solo inf p50/p95={r['solo_inferencia_p50_ms']}/"
              f"{r['solo_inferencia_p95_ms']} ms | RAM {r['ram_proceso_mb']} MB (pico cgroup "
              f"{r['ram_contenedor_pico_mb']}) | frio {r['arranque_frio_ms']} ms")

    mem_limit = None
    p = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
    if os.path.exists(p) and int(open(p).read()) < 1 << 50:
        mem_limit = round(int(open(p).read()) / 2**30, 2)
    flags = open("/proc/cpuinfo").read() if os.path.exists("/proc/cpuinfo") else ""
    cpu_name = next((ln.split(":", 1)[1].strip() for ln in flags.splitlines() if ln.startswith("model name")),
                    platform.processor() or platform.machine())
    hw = {"cpu": cpu_name, "vcpu_visibles": os.cpu_count(), "hilos_usados": a.threads,
          "limite_memoria_gb": mem_limit, "avx512_vnni": "avx512_vnni" in flags,
          "so": platform.platform(), "python": platform.python_version(), "nota": a.hardware_note}
    Path(a.out).parent.mkdir(exist_ok=True)
    if a.append and Path(a.out).exists():
        old = json.load(open(a.out, encoding="utf-8"))["resultados"]
        results = [r for r in old if r["backend"] not in {x["backend"] for x in results}] + results
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"hardware": hw, "resultados": results}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
