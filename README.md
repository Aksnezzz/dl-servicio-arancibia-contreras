# dl-servicio-arancibia-contreras

Servicio de inferencia de Deep Learning para clasificar prendas de vestir (Caso A – Visión).
Asignatura Machine Learning (TIEL26). Equipo: **Francisco Arancibia** y **Benjamín Contreras**.

> Estado: **Hito 1 (avance parcial)** — baseline entrenado, exportado a ONNX, cuantizado a INT8 y medido.
> La API FastAPI, el Dockerfile y el workflow de GitHub Actions se agregan en la semana 2 (ver plan de acción).

## Problema

Una tienda de vestuario de la Región de Coquimbo necesita clasificar automáticamente fotos de productos en
10 categorías (camiseta, pantalón, suéter, vestido, abrigo, sandalia, camisa, zapatilla, bolso, botín) para
ordenar su catálogo. Requisito del cliente: < 300 ms por petición en CPU con ≤ 4 GB de RAM.

## Modelo baseline

- **Arquitectura:** MobileNetV2 preentrenada en ImageNet, se reemplaza la última capa por una lineal de 10 clases y se ajusta (fine-tuning) toda la red.
- **Datos:** Fashion-MNIST (60.000 train / 10.000 test, 28×28 gris). Se reescala a 96×96 y se replica a 3 canales.
- **Entrenamiento:** 12.000 imágenes, 2 épocas, AdamW + OneCycle (lr 1e-3), label smoothing 0,1, semilla 42. ~4 min en 2 vCPU.

## Resultados preliminares (2 vCPU Intel Xeon 2,1 GHz, 300 peticiones de 1 imagen)

| Métrica | PyTorch FP32 | ONNX Runtime FP32 | ONNX Runtime INT8 |
|---|---|---|---|
| Exactitud (test, 10.000) | 0,9190 | 0,9190 | 0,9099 |
| F1 macro | 0,9191 | 0,9191 | 0,9110 |
| Tamaño del artefacto | 9,19 MB | 8,92 MB | 2,61 MB |
| Latencia p50 | 6,55 ms | 1,21 ms | 1,10 ms |
| Latencia p95 | 10,88 ms | 1,64 ms | 1,78 ms |
| Throughput secuencial | 130,5 pet/s | 746,9 pet/s | 800,2 pet/s |
| RAM del proceso | 971,7 MB | 211,9 MB | 191,8 MB |
| Arranque en frío | 2.992 ms | 68 ms | 74 ms |

Detalle en `reports/baseline_metrics.json` y `reports/benchmark.json`. La latencia es de inferencia en proceso
(preprocesamiento + modelo); la latencia HTTP de punta a punta se medirá en el Hito 2.

## Cómo reproducir

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Datos (Fashion-MNIST, ~30 MB)
mkdir -p data/fashion
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -L -o data/fashion/$f.gz https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/$f.gz
done

# 2. Entrenar baseline (descarga pesos ImageNet de torchvision)
python scripts/train_baseline.py

# 3. Exportar a ONNX + cuantizar INT8
python scripts/export_onnx.py

# 4. Medir latencia, throughput, RAM y exactitud
python scripts/benchmark.py --requests 300
```

También se puede ejecutar todo desde `notebooks/01_baseline.ipynb` (Google Colab).

## Estructura

```
app/preprocessing.py        # preprocesamiento sin PyTorch (lo usará la API)
notebooks/01_baseline.ipynb # entrenamiento y métricas
scripts/train_baseline.py   # entrenamiento reproducible
scripts/export_onnx.py      # exportación ONNX + cuantización estática INT8
scripts/benchmark.py        # p50/p95, throughput, RAM, arranque en frío
reports/                    # métricas en JSON
model/                      # artefactos (baseline.pt, model_fp32.onnx, model_int8.onnx)
```

Pendiente semana 2: `app/main.py` (FastAPI `/health`, `/predict`), `tests/`, `Dockerfile`, `.github/workflows/ci.yml`.

## Uso de IA generativa

Se utilizó Claude (Anthropic) como apoyo para estructurar los scripts y redactar documentación.
Todo el código fue revisado y puede ser explicado por ambos integrantes.
