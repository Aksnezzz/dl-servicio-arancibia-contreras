"""Entrenamiento del modelo baseline (Caso A - Vision).

Transfer learning con MobileNetV2 preentrenado en ImageNet sobre Fashion-MNIST
(10 clases de prendas). Guarda el modelo en model/baseline.pt y las metricas
en reports/baseline_metrics.json.

Uso:
    python scripts/train_baseline.py                       # descarga pesos de torchvision
    python scripts/train_baseline.py --weights-path pesos.pth  # pesos ImageNet locales
"""
import argparse
import gzip
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import classification_report, confusion_matrix, f1_score

CLASSES = ["camiseta", "pantalon", "sueter", "vestido", "abrigo",
           "sandalia", "camisa", "zapatilla", "bolso", "botin"]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.preprocessing import preprocess as svc_preprocess  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    if int.from_bytes(data[2:3], "big") != 8:
        raise ValueError("formato IDX inesperado")
    ndim = data[3]
    dims = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load_fashion(split: str):
    d = ROOT / "data" / "fashion"
    prefix = "train" if split == "train" else "t10k"
    x = load_idx(d / f"{prefix}-images-idx3-ubyte.gz")
    y = load_idx(d / f"{prefix}-labels-idx1-ubyte.gz")
    return torch.from_numpy(x.copy()), torch.from_numpy(y.astype(np.int64))


def preprocess(x_uint8: torch.Tensor, size: int) -> torch.Tensor:
    """(N,28,28) uint8 -> (N,3,size,size) float normalizado como ImageNet."""
    x = x_uint8.float().div(255.0).unsqueeze(1)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    return (x - MEAN) / STD


def build_model(weights_path: str | None) -> nn.Module:
    if weights_path:
        model = torchvision.models.mobilenet_v2()
        sd = torch.load(weights_path, map_location="cpu", weights_only=True)
        sd = sd.get("state_dict", sd)
        target = model.state_dict()
        # mapeo por orden (misma arquitectura, nombres pueden variar)
        mapped = {k: v for (k, _), v in zip(target.items(), sd.values())}
        model.load_state_dict(mapped)
    else:
        w = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
        model = torchvision.models.mobilenet_v2(weights=w)
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))
    return model


@torch.no_grad()
def evaluate(model, x_uint8: np.ndarray, y: np.ndarray, size: int, bs: int = 250):
    """Evalua con el MISMO preprocesamiento que usa el servicio (app/preprocessing.py, Pillow)."""
    model.eval()
    preds = [model(torch.from_numpy(svc_preprocess(x_uint8[i:i + bs], size))).argmax(1).numpy()
             for i in range(0, len(x_uint8), bs)]
    p = np.concatenate(preds)
    return {"accuracy": round(float((p == y).mean()), 4),
            "f1_macro": round(float(f1_score(y, p, average="macro")), 4)}, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-path", default=None)
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--train-samples", type=int, default=12000)
    ap.add_argument("--val-samples", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-only", action="store_true", help="no entrena: evalua model/baseline.pt")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    xall, yall = load_fashion("train")
    xte, yte = load_fashion("test")
    # Particion reproducible del set de entrenamiento oficial (60.000):
    #   entrenamiento = 12.000 | validacion = 5.000 (nunca vistas al entrenar) | resto sin usar
    perm = torch.randperm(len(xall))
    tr_idx = perm[: args.train_samples]
    va_idx = perm[args.train_samples: args.train_samples + args.val_samples]
    xtr, ytr = xall[tr_idx], yall[tr_idx]
    xva, yva = xall[va_idx].numpy(), yall[va_idx].numpy()

    (ROOT / "model").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    prev = {}
    if (ROOT / "reports" / "baseline_metrics.json").exists():
        prev = json.load(open(ROOT / "reports" / "baseline_metrics.json", encoding="utf-8"))

    if args.eval_only:
        model = torchvision.models.mobilenet_v2(num_classes=len(CLASSES))
        model.load_state_dict(torch.load(ROOT / "model" / "baseline.pt", map_location="cpu", weights_only=True))
        train_time = prev.get("tiempo_entrenamiento_s")
    else:
        model = build_model(args.weights_path)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        steps = args.epochs * ((len(xtr) + args.batch_size - 1) // args.batch_size)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        t0 = time.time()
        for ep in range(args.epochs):
            model.train()
            order = torch.randperm(len(xtr))
            total, n = 0.0, 0
            for i in range(0, len(xtr), args.batch_size):
                b = order[i:i + args.batch_size]
                xb = xtr[b]
                if random.random() < 0.5:  # aumento simple: volteo horizontal
                    xb = torch.flip(xb, dims=[2])
                out = model(preprocess(xb, args.img_size))
                loss = loss_fn(out, ytr[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                total += loss.item() * len(b)
                n += len(b)
            print(f"epoca {ep + 1}/{args.epochs} - loss {total / n:.4f} - {time.time() - t0:.0f}s", flush=True)
        train_time = round(time.time() - t0, 1)
        model.eval()
        torch.save(model.state_dict(), ROOT / "model" / "baseline.pt")

    val_m, _ = evaluate(model, xva, yva, args.img_size)
    test_m, p = evaluate(model, xte.numpy(), yte.numpy(), args.img_size)
    y = yte.numpy()
    print(f"validacion: {val_m} | prueba: {test_m}")

    # subconjunto fijo de validacion para la prueba de calidad en CI (sin PyTorch)
    (ROOT / "tests" / "data").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ROOT / "tests" / "data" / "val_subset_500.npz", x=xva[:500], y=yva[:500])
    with open(ROOT / "reports" / "particion.json", "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "fuente": "train-images-idx3-ubyte (60.000)",
                   "entrenamiento": tr_idx.tolist(), "validacion": va_idx.tolist()}, f)

    metrics = {
        "modelo": "MobileNetV2 (ImageNet) + cabeza lineal 10 clases",
        "dataset": "Fashion-MNIST",
        "entrada": (f"28x28 gris -> {args.img_size}x{args.img_size} bilineal, "
                    "replicada a 3 canales, normalizacion ImageNet"),
        "train_samples": len(xtr),
        "val_samples": len(xva),
        "test_samples": len(xte),
        "img_size": args.img_size,
        "epochs": args.epochs,
        "seed": args.seed,
        "validacion": val_m,
        "prueba": test_m,
        "accuracy": test_m["accuracy"],
        "f1_macro": test_m["f1_macro"],
        "tamano_mb": round(os.path.getsize(ROOT / "model" / "baseline.pt") / 1e6, 2),
        "tiempo_entrenamiento_s": train_time,
        "hardware": f"{platform.processor() or platform.machine()} - {os.cpu_count()} vCPU",
        "por_clase": classification_report(y, p, target_names=CLASSES, output_dict=True, zero_division=0),
        "matriz_confusion": confusion_matrix(y, p).tolist(),
    }
    with open(ROOT / "reports" / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("por_clase", "matriz_confusion")}, indent=2))


if __name__ == "__main__":
    main()
