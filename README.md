# dl-servicio-arancibia-contreras

Servicio de inferencia de Deep Learning para clasificar prendas de vestir (Caso A – Visión).
Asignatura Machine Learning (TIEL26). Equipo: **Francisco Arancibia** y **Benjamín Contreras**.

> Estado: **Hito 1 (avance parcial)**: baseline entrenado, exportado a ONNX, cuantizado a INT8 y medido
> dentro de un contenedor limitado a 4 GB de RAM y 2 CPU. La API FastAPI, el Dockerfile y el workflow de
> GitHub Actions se agregan en la semana 2 (ver plan de acción).

## Problema

Una tienda de vestuario de la Región de Coquimbo necesita clasificar automáticamente fotos de productos en
10 categorías (camiseta, pantalón, suéter, vestido, abrigo, sandalia, camisa, zapatilla, bolso, botín) para
ordenar su catálogo. Requisito del cliente: < 300 ms por petición en CPU con ≤ 4 GB de RAM.

## Modelo baseline

- **Arquitectura:** MobileNetV2 preentrenada en ImageNet; se reemplaza la última capa por una lineal de 10 clases y se ajusta toda la red.
- **Entrada:** 28×28 en gris → 96×96 (bilineal) → replicada a 3 canales → normalización ImageNet.
- **Particiones** (`reports/particion.json`): 12.000 imágenes de entrenamiento y 5.000 de validación, ambas
  del set de entrenamiento oficial, y las 10.000 de prueba oficial.
- **Entrenamiento:** 2 épocas, AdamW + OneCycle (lr 1e-3), label smoothing 0,1, volteo horizontal, semilla 42. Toma ~4 min en 2 vCPU.
- **Optimización:** exportación a ONNX y cuantización estática INT8 por canal (QDQ), calibrada con 300 imágenes de entrenamiento.

## Resultados (contenedor Docker `--memory=4g --cpus=2`, Intel Xeon 2,1 GHz con AVX-512 VNNI)

Flujo real de `/predict`: una foto JPEG de 3000×3000 (~1,8 MB) se decodifica a escala reducida, se pasa a gris,
se recorta la prenda, se lleva a 28×28 → 96×96×3 y se infiere. Se usaron 300 peticiones de 1 imagen, tras 20 de calentamiento.

| Métrica | PyTorch FP32 | ONNX FP32 | ONNX INT8 |
|---|---|---|---|
| Exactitud / F1 macro en prueba (10.000) | 0,919 / 0,919 | 0,919 / 0,919 | 0,911 / 0,912 |
| Exactitud en validación (5.000) | 0,918 | 0,918 | 0,914 |
| Tamaño del artefacto | 9,19 MB | 8,92 MB | 2,61 MB |
| **Foto JPEG: latencia total p50 / p95** | **33,8 / 42,4 ms** | **26,1 / 32,0 ms** | **25,1 / 30,4 ms** |
| ↳ decodificación y adaptación de la foto (p50) | 24,7 ms | 24,4 ms | 24,0 ms |
| ↳ preprocesamiento 96×96×3 (p50) | 0,13 ms | 0,14 ms | 0,12 ms |
| ↳ inferencia (p50) | 8,35 ms | 1,58 ms | 1,03 ms |
| Solo inferencia p50 / p95 | 6,00 / 8,26 ms | 1,05 / 1,44 ms | 0,69 / 0,87 ms |
| Throughput (foto / solo inferencia) | 29 / 158 pet/s | 37 / 874 pet/s | 39 / 1.403 pet/s |
| RAM del proceso (RSS) | 813 MB | 139 MB | 130 MB |
| Arranque del proceso (importar + cargar + 1.ª predicción) | 3.095 ms | 244 ms | 216 ms |

Las fotos de `scripts/make_test_jpegs.py` son **sintéticas**: sirven para medir tiempos, no exactitud.
La decodificación del JPEG ocupa ~95 % del tiempo total. Con `Image.draft()` baja de 61 a 23,5 ms.
Detalle en `reports/benchmark.json`, `reports/evaluation.json` y `reports/baseline_metrics.json`.

## Cómo reproducir

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Datos (Fashion-MNIST, ~30 MB)
mkdir -p data/fashion
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -L -o data/fashion/$f.gz https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/$f.gz
done

# 2. Entrenar el baseline (descarga los pesos ImageNet de torchvision) — o solo evaluar: --eval-only
python scripts/train_baseline.py

# 3. Exportar a ONNX + cuantizar INT8
python scripts/export_onnx.py

# 4. Exactitud de los modelos ONNX (verifica CA1: >= 0,90 en prueba)
python scripts/evaluate.py

# 5. Fotos sintéticas y benchmark por etapas
python scripts/make_test_jpegs.py --n 20
python scripts/benchmark.py --requests 300
```

### Benchmark con los límites del cliente (4 GB, 2 CPU)

Se usa un contenedor por modelo, para que la RAM de uno no afecte a otro:

```bash
for b in pytorch_fp32 onnx_fp32 onnx_int8; do
  docker run --rm --memory=4g --memory-swap=4g --cpus=2 -v "$PWD":/work -w /work python:3.12-slim \
    sh -c "pip install -q -r requirements.txt && python scripts/benchmark.py --backends $b --append"
done
```

## Estructura

```
app/preprocessing.py         # preprocesamiento sin PyTorch (lo usará la API): foto -> 28x28 -> 96x96x3
notebooks/01_baseline.ipynb  # entrenamiento y métricas (Colab)
scripts/train_baseline.py    # entrenamiento reproducible + particiones + métricas
scripts/export_onnx.py       # exportación ONNX + cuantización estática INT8
scripts/evaluate.py          # exactitud/F1 de los ONNX en validación, prueba y subconjunto de CI
scripts/make_test_jpegs.py   # fotos JPEG sintéticas de ~1,8 MB para medir tiempos
scripts/benchmark.py         # p50/p95 por etapa, throughput, RAM, arranque
tests/data/val_subset_500.npz# 500 imágenes de validación para la prueba de calidad en CI (umbral 0,87)
reports/                     # métricas en JSON y figuras
model/model_int8.onnx        # artefacto optimizado (2,6 MB)
```

Pendiente para la semana 2: `app/main.py` (FastAPI `/health`, `/predict`), `tests/`, `Dockerfile` y `.github/workflows/ci.yml`.

## Uso de IA generativa

Se utilizó Claude (Anthropic) como apoyo para estructurar los scripts y redactar documentación.
Todo el código fue revisado y ejecutado, y puede ser explicado por ambos integrantes.
