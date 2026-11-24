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


def image_bytes_to_array(img: Image.Image) -> np.ndarray:
    """Imagen arbitraria (foto de producto) -> arreglo 28x28 estilo Fashion-MNIST."""
    g = img.convert("L").resize((28, 28), Image.BILINEAR)
    arr = np.asarray(g, dtype=np.uint8)
    if arr.mean() > 127:  # Fashion-MNIST: prenda clara sobre fondo negro
        arr = 255 - arr
    return arr
