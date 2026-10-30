# TP1 - Sistema de Reconocimiento Facial

Plantilla base para desarrollar un sistema completo de detección, alineación, extracción de embeddings e identificación/verificación facial.

> 📄 El enunciado completo del trabajo práctico está en [ENUNCIADO.md](ENUNCIADO.md).

## Objetivo del backend

Implementar una API asincrónica en Python que permita:

- Registrar identidades (`/insert`)
- Ejecutar inferencia sobre una imagen (`/predict`)
- Consultar el estado del procesamiento asincrónico (`/status/{job_id}`)

La API responde `HTTP 202` con un `job_id` y luego permite consultar el resultado con estado:

```json
{
  "status": "done | inProgress | failed",
  "link": "url | none"
}
```

## Estructura

```text
tuia-face-recognition-app/
├── 1-pre-processing.ipynb       # Parte 1: dataset y preprocesamiento
├── 2-train.ipynb                # Parte 2: entrenamiento y comparación de modelos
├── src/
│   ├── app/
│   │   └── main.py              # App FastAPI + /health
│   ├── frontend/                # UI Gradio provista por la cátedra
│   │   ├── app.py
│   │   └── gradio_ui.py
│   └── lib/
│       ├── api.py               # Endpoints /upload, /insert, /predict, /status, /files
│       ├── config.py            # Configuración vía .env
│       ├── schemas.py
│       ├── services/
│       │   ├── face_service.py  #
│       │   └── task_manager.py
│       └── storage/
│           ├── base.py
│           ├── embedding_store.py   # Persistencia en JSON
│           └── pgvector_store.py    # Persistencia en PostgreSQL + pgvector
├── models/                      # Modelo entrenado (.pth / .onnx), no se versiona
├── output/                      # Imágenes subidas y resultados, no se versiona
├── requirements.txt
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── .env.docker.example
└── .env.local.example
```

# Preparando el ambiente local

## Requisitos para trabajar de forma local

- Python 3.12
- Docker

## Configurá tu modelo

Entrená tu modelo y guardalo dentro de la carpeta `models`. El backend admite modelos con extensión **.pth** (PyTorch, se carga con `torch.load` el modelo completo) o **.onnx** (se carga con `onnxruntime`). Si usan otro framework, expórtenlo a **.onnx**.

Recordá actualizar la configuración del `.env` correspondiente (`MODEL_NAME`, `EMBEDDING_DIM`, `FACE_SIZE`, `SIMILARITY_THRESHOLD`) para que apunte a tu modelo.

> El backend no arranca si no existe el archivo `models/<MODEL_NAME>`.

Con docker compose, el entorno reinicia la aplicación y actualiza el código automáticamente al detectar cambios. Lo mismo ocurre en local si usan `--reload`.

El reinicio automático puede no funcionar en algunas versiones de Docker Desktop para _Windows_. En ese caso, vuelvan a correr los comandos del siguiente apartado para actualizar el código dentro de Docker.

## Opción 1 - Corriendo dentro de Docker

### 1. Construí y corré la aplicación

Ajustá el archivo **.env.docker.example** a tus necesidades. Luego corré desde la terminal:

```bash
docker compose build
docker compose up -d
```

## Opción 2 - Configurando el ambiente local

### 1. Instalá uv

Para usuarios de Linux o Mac:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Para usuarios de Windows:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

[Documentación de uv](https://docs.astral.sh/uv/getting-started/installation/).

### 2. Configurá un ambiente virtual con Python 3.12

```bash
uv venv --python 3.12 .venv
```

### 3. Activá el ambiente virtual

```bash
source .venv/bin/activate
```

### 4. Instalá las dependencias

```bash
uv pip install -r requirements.txt
```

### 5. Iniciá la base de datos

```bash
docker compose up postgres -d
```

### 6. Iniciá el frontend

```bash
cd src
uvicorn frontend.app:app --port 8080
```

### 7. Iniciá el backend

Ajustá el archivo _.env.local.example_ a tus necesidades. Desde la carpeta `src`, en otra terminal:

```bash
cp ../models/<NOMBRE_DE_TU_MODELO> models/
cp ../.env.local.example .env
uvicorn app.main:app --reload --port 8000
```

## Configuración

No hardcodeen parámetros: configúrenlos mediante `.env`.

1. Para la ejecución local, copien `.env.local.example` a `src/.env`.
2. Ajusten las variables del modelo, los paths y el threshold.
3. Configuren el detector: `DETECTOR_MODEL_NAME` (opcional, archivo `.pth`/`.onnx` en `models/`), `DET_SCORE_THRESHOLD`, `DET_INPUT_SIZE`, `MIN_FACE_SIZE` y `ALIGN_MARGIN`. Si su detector viene de una librería, sobrescriban `_load_detector` en `face_service.py`.
4. Opcional (habilitado por defecto): configuren la conexión a PostgreSQL + pgvector (`USE_PGVECTOR`). Con `USE_PGVECTOR=false`, los embeddings se guardan en un archivo JSON (`EMBEDDINGS_PATH`).

## Endpoints

- Backend: `http://localhost:8000` (documentación Swagger en `http://localhost:8000/docs`)
- PostgreSQL/pgvector: `localhost:5432`
- Frontend (provisto por la cátedra): `http://localhost:8080`

## Pipeline

## Pipeline

### Provisto por la cátedra

- API asincrónica (`/upload`, `/insert`, `/predict`, `/status`) y manejo de jobs.
- Carga del modelo (`.pth` / `.onnx`) desde `models/`.
- Búsqueda por similitud configurable (`cosine` o `l2`) e identificación (`identify`).
- Manejo de desconocidos con `SIMILARITY_THRESHOLD`.
- Persistencia configurable en JSON o en PostgreSQL + pgvector (`USE_PGVECTOR`).
- Frontend para registrar identidades, predecir y visualizar los resultados (cajas, etiquetas y keypoints).

### Implementado por el grupo (`src/lib/services/face_service.py`)

1. `detect_faces`: detector SCRFD (`det_10g.onnx`) con un borde configurable (`DET_PAD`) alrededor de la imagen. Filtra por score (`DET_SCORE_THRESHOLD`) y por tamaño mínimo (`MIN_FACE_SIZE`).
2. `align_face`: recorta alrededor de la caja con margen (`ALIGN_MARGIN`), busca los 5 landmarks y alinea con `norm_crop` a `FACE_SIZE`. Devuelve los keypoints relativos a la caja.
3. `extract_embedding_from_face`: ArcFace (`w600k_r50.onnx`) con el mismo preprocesamiento que en las notebooks (BGR→RGB, (x−127,5)/127,5, HWC→CHW). Devuelve el embedding normalizado (L2).

Se agregó la variable `DET_PAD` en `config.py`, `api.py` y los `.env.*.example`, para que el borde del detector sea configurable.

## Modelo y fine-tuning

**Modelo elegido:** ArcFace preentrenado de InsightFace (paquete `buffalo_l`), usado sin reentrenar.

- **Reconocedor:** `w600k_r50.onnx`, una ResNet-50 entrenada con la loss ArcFace sobre WebFace600K. Recibe caras alineadas de 112×112 y devuelve embeddings de 512 valores.
- **Detector:** `det_10g.onnx` (SCRFD), que devuelve la caja y los 5 landmarks de cada cara.
- **Descarga (link público):** https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip. Del zip se usan solo `w600k_r50.onnx` y `det_10g.onnx`, que van en la carpeta `models/`.
- **Licencia:** los modelos de InsightFace son solo para uso no comercial o de investigación.

**Modelo descartado:** una CNN propia de 4 bloques convolucionales (433 mil parámetros, embedding de 128), entrenada desde cero con clasificación + embedding sobre 85 personas. Hiperparámetros: Adam (lr 0,001, weight decay 1e-4), ReduceLROnPlateau, batch 64, 30 épocas con early stopping (paciencia 5) y augmentation con albumentations. El detalle está en `2-train.ipynb`.

**Comparación (test con personas no vistas):**

| Métrica                         | CNN propia            | ArcFace               |
| ------------------------------- | --------------------- | --------------------- |
| AUC (verificación)              | 0,850                 | 1,000                 |
| EER                             | 0,236                 | 0,002                 |
| TAR @ FAR=1e-3                  | 0,051                 | 0,997                 |
| Top-1 accuracy                  | 0,433                 | 0,999                 |
| Precision / Recall / F1 (macro) | 0,440 / 0,051 / 0,084 | 1,000 / 0,993 / 0,996 |
| Desconocidos rechazados         | 0,975                 | 0,999                 |
| Tiempo por cara en CPU          | 9 ms                  | 104 ms                |
| Tamaño                          | 1,7 MB                | 174 MB                |

**Justificación:** ArcFace es muy superior en todas las métricas, y su costo (104 ms por cara en CPU) es aceptable para la app. La CNN rinde mucho menos porque la entrenamos con pocas identidades y pocas imágenes, es 100 veces más chica, usa una loss de clasificación que no fuerza a separar bien los embeddings, y no terminó de converger en 30 épocas en CPU. Es probable que algunas celebridades de Pins estuvieran en los datos de entrenamiento de ArcFace, así que sus resultados pueden estar algo inflados.

**Umbral:** `SIMILARITY_THRESHOLD=0.33` con métrica coseno. Lo calibramos con validación, simulando la identificación: es el umbral más bajo que rechaza al 99% de las personas no registradas. Con él, en test se identificó correctamente al 99,3% de las personas registradas y se rechazó al 99,9% de las desconocidas.

**Análisis de errores:** los pocos errores de ArcFace se deben a diferencias grandes de edad (fotos de la infancia), condiciones extremas (blanco y negro, anteojos oscuros, ojos cerrados), oclusiones (máscaras) y una ilustración que se había colado en el dataset.

## Dataset

- **Fuente:** Pins Face Recognition (Kaggle), fotos de 105 celebridades sacadas de Pinterest. Licencia: \_\_\_\_. Se usó solo con fines académicos y no se incluye en el repositorio.
- **Tamaño original:** 17.499 imágenes, entre 86 y 237 por persona (media 167), con un desbalance moderado.
- **Variaciones:** mucha variación de iluminación, pose y expresión; algunas fotos en blanco y negro, con maquillaje de personaje o con oclusiones.
- **Detección y alineación:** SCRFD con un borde del 50% alrededor de la imagen (sin ese borde el detector fallaba en el 90% de los casos, porque las fotos ya vienen recortadas alrededor de la cara) y alineación con los 5 landmarks (`norm_crop`) a 112×112.
- **Filtrado:** imágenes corruptas, sin cara o con más de una, caras chicas, borrosas (varianza del Laplaciano), con exposición extrema, duplicados y casi-duplicados (dHash), posibles errores de etiquetado (similitud baja con su propia identidad) y casi-duplicados entre train y test (similitud > 0,95).
- **Resultado final:** 16.650 caras alineadas.

| Split                          | Personas | Imágenes |
| ------------------------------ | -------- | -------- |
| train                          | 85       | 9.581    |
| val                            | 85       | 1.978    |
| test_closed                    | 85       | 1.986    |
| test_open (personas no vistas) | 20       | 3.105    |

## Notas importantes

- Para usar `pgvector`, levanten `postgres` y definan `USE_PGVECTOR=true` en `.env`.
- Si cambian `EMBEDDING_DIM`, la tabla `embeddings` se recrea y se borran los registros previos.
- Los modelos (`*.pth`, `*.onnx`) y los datos no se versionan (ver `.gitignore`). Coloquen el modelo entrenado en `models/` y súbanlo a un link **público de solo lectura** para que los docentes puedan descargarlo.
