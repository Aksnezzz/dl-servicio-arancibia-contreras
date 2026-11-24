"""Evalua exactitud y F1 macro de los modelos ONNX (verificador del criterio CA1).

Conjuntos: validacion (5.000 imagenes del set de entrenamiento oficial que el modelo no vio,
indices en reports/particion.json), prueba oficial (10.000) y subconjunto de CI (500 de validacion).
No requiere PyTorch.

Uso: python scripts/evaluate.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.preprocessing import load_fashion, preprocess  # noqa: E402

CA1_UMBRAL = 0.90


def score(sess, x, y, bs=50):
    p = np.concatenate([sess.run(None, {"input": preprocess(x[i:i + bs])})[0].argmax(1)
                        for i in range(0, len(x), bs)])
    acc = float((p == y).mean())
    return {"accuracy": round(acc, 4), "f1_macro": round(float(f1_score(y, p, average="macro")), 4),
            "error_estandar": round(float(np.sqrt(acc * (1 - acc) / len(y))), 4), "n": int(len(y))}


def main():
    xtr, ytr = load_fashion(ROOT, "train")
    xte, yte = load_fashion(ROOT, "test")
    va = np.array(json.load(open(ROOT / "reports" / "particion.json"))["validacion"])
    ci = np.load(ROOT / "tests" / "data" / "val_subset_500.npz")
    out = {}
    for name in ("model_fp32.onnx", "model_int8.onnx"):
        sess = ort.InferenceSession(str(ROOT / "model" / name), providers=["CPUExecutionProvider"])
        out[name] = {"validacion_5000": score(sess, xtr[va], ytr[va]),
                     "prueba_10000": score(sess, xte, yte),
                     "subconjunto_ci_500": score(sess, ci["x"], ci["y"])}
        r = out[name]
        ok = "CUMPLE" if r["prueba_10000"]["accuracy"] >= CA1_UMBRAL else "NO CUMPLE"
        print(f"{name:16s} val={r['validacion_5000']['accuracy']:.4f} test={r['prueba_10000']['accuracy']:.4f} "
              f"ci500={r['subconjunto_ci_500']['accuracy']:.4f}  CA1 (>= {CA1_UMBRAL}): {ok}")
    with open(ROOT / "reports" / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
