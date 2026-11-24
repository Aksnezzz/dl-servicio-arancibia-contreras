"""Preprocesamiento sin dependencia de PyTorch (lo usan la API y el benchmark)."""
import gzip
from pathlib import Path

import numpy as np
from PIL import Image

IMG_SIZE = 96
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
CLASSES = ["camiseta", "pantalon", "sueter", "vestido", "abrigo",
           "sandalia", "camisa", "zapatilla", "bolso", "botin"]


def load_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    dims = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load_fashion(root: Path, split: str):
    prefix = "train" if split == "train" else "t10k"
    d = root / "data" / "fashion"
    return (load_idx(d / f"{prefix}-images-idx3-ubyte.gz"),
            load_idx(d / f"{prefix}-labels-idx1-ubyte.gz").astype(np.int64))


def preprocess(images: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """(N,H,W) uint8 en escala de grises -> (N,3,size,size) float32 normalizado."""
    out = np.empty((len(images), 1, size, size), dtype=np.float32)
    for i, im in enumerate(images):
        out[i, 0] = np.asarray(Image.fromarray(im).resize((size, size), Image.BILINEAR), dtype=np.float32)
    out = np.repeat(out / 255.0, 3, axis=1)
    return ((out - MEAN) / STD).astype(np.float32)


def image_bytes_to_array(img: Image.Image, work_size: int = 256) -> np.ndarray:
    """Foto de producto -> arreglo 28x28 al estilo Fashion-MNIST.

    1. draft(): en JPEG decodifica directamente a escala reducida (1/2, 1/4 o 1/8),
       mucho mas rapido que decodificar 12 MP completos.
    2. Escala de grises e inversion si el fondo es claro (Fashion-MNIST: prenda clara sobre negro).
    3. Recorta a la prenda (caja que contiene los pixeles de primer plano), la centra en un
       cuadrado con margen y reescala a 28x28.
    """
    if img.format == "JPEG":
        img.draft("L", (work_size, work_size))
    g = img.convert("L")
    g.thumbnail((work_size, work_size), Image.BILINEAR)
    arr = np.asarray(g, dtype=np.uint8)
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    if np.median(border) > 127:
        arr = 255 - arr
    fg = arr > max(40, int(np.median(border if np.median(border) <= 127 else 255 - border)) + 30)
    if fg.any():
        rows, cols = np.where(fg.any(1))[0], np.where(fg.any(0))[0]
        arr = arr[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    h, w = arr.shape
    side = int(max(h, w) * 1.15)  # margen ~7 % por lado, como en Fashion-MNIST
    canvas = np.zeros((side, side), dtype=np.uint8)
    canvas[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = arr
    return np.asarray(Image.fromarray(canvas).resize((28, 28), Image.BILINEAR), dtype=np.uint8)
