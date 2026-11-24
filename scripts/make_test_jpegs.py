"""Genera fotos JPEG SINTETICAS de ~2 MB para medir la latencia del flujo real de /predict
(decodificacion + preprocesamiento + inferencia).

Cada foto toma una prenda del set de prueba, la amplia a ~3000x3000 px y la pone, invertida,
sobre un fondo claro con textura y ruido (para que el JPEG pese lo que pesa una foto de celular).
Sirven para medir TIEMPOS; la exactitud con fotos reales se mide con las fotos propias del equipo.

Uso: python scripts/make_test_jpegs.py --n 20
"""
import argparse
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.preprocessing import load_fashion  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--side", type=int, default=3000)
    ap.add_argument("--out", default=str(ROOT / "data" / "jpeg_sinteticas"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    x, y = load_fashion(ROOT, "test")
    rng = np.random.default_rng(7)
    labels = []
    for k, i in enumerate(rng.choice(len(x), a.n, replace=False)):
        prenda = Image.fromarray(255 - x[i]).resize((int(a.side * 0.8),) * 2, Image.BICUBIC)
        fondo = rng.normal(235, 12, (a.side, a.side)).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(fondo).filter(ImageFilter.GaussianBlur(1))
        mask = Image.fromarray(x[i]).resize(prenda.size, Image.BICUBIC).point(lambda v: 255 if v > 20 else 0)
        off = int(a.side * 0.1)
        img.paste(prenda, (off, off), mask)
        ruido = rng.normal(0, 6, (a.side, a.side))
        img = Image.fromarray((np.asarray(img, dtype=np.float32) + ruido).clip(0, 255).astype(np.uint8)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        (out / f"foto_{k:02d}.jpg").write_bytes(buf.getvalue())
        labels.append(int(y[i]))
    np.save(out / "labels.npy", np.array(labels))
    sizes = [p.stat().st_size / 1e6 for p in sorted(out.glob("*.jpg"))]
    print(f"{len(sizes)} JPEG en {out} - tamano medio {np.mean(sizes):.2f} MB "
          f"(min {min(sizes):.2f}, max {max(sizes):.2f})")


if __name__ == "__main__":
    main()
