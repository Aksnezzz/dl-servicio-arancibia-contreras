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
def evaluate(model, x, y, size, bs=256):
    model.eval()
    preds = []
    for i in range(0, len(x), bs):
        preds.append(model(preprocess(x[i:i + bs], size)).argmax(1))
    p = torch.cat(preds)
    return p.numpy(), y.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-path", default=None)
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--train-samples", type=int, default=12000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    xtr, ytr = load_fashion("train")
    xte, yte = load_fashion("test")
    idx = torch.randperm(len(xtr))[: args.train_samples]
    xtr, ytr = xtr[idx], ytr[idx]

    model = build_model(args.weights_path)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(xtr) + args.batch_size - 1) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xtr))
        total, n = 0.0, 0
        for i in range(0, len(xtr), args.batch_size):
            b = perm[i:i + args.batch_size]
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
    train_time = time.time() - t0

    p, y = evaluate(model, xte, yte, args.img_size)
    acc = float((p == y).mean())
    f1 = float(f1_score(y, p, average="macro"))
    print(f"test accuracy={acc:.4f}  F1 macro={f1:.4f}")

    (ROOT / "model").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    model.eval()
    torch.save(model.state_dict(), ROOT / "model" / "baseline.pt")
    size_mb = os.path.getsize(ROOT / "model" / "baseline.pt") / 1e6

    metrics = {
        "modelo": "MobileNetV2 (ImageNet) + cabeza lineal 10 clases",
        "dataset": "Fashion-MNIST",
        "train_samples": len(xtr),
        "test_samples": len(xte),
        "img_size": args.img_size,
        "epochs": args.epochs,
        "seed": args.seed,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
        "tamano_mb": round(size_mb, 2),
        "tiempo_entrenamiento_s": round(train_time, 1),
        "hardware": f"{platform.processor() or platform.machine()} - {os.cpu_count()} vCPU",
        "por_clase": classification_report(y, p, target_names=CLASSES, output_dict=True, zero_division=0),
        "matriz_confusion": confusion_matrix(y, p).tolist(),
    }
    with open(ROOT / "reports" / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("por_clase", "matriz_confusion")}, indent=2))


if __name__ == "__main__":
    main()
