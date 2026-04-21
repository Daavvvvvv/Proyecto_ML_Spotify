# Proyecto ML — Clasificación de género musical en Spotify

Proyecto integrador del curso **Aprendizaje de Máquina Aplicado** (ST1613/ST1631, EAFIT, 2026-1).
Profesor: Marco Teran (`mtteranl@eafit.edu.co`).

## Equipo

- [TODO: integrante 1]
- [TODO: integrante 2]
- [TODO: integrante 3]

Equipo de 3 personas autorizado explícitamente por el profesor (fuera del default individual/parejas).

## Problema

**Pregunta:** ¿Se puede predecir el género musical de una canción a partir únicamente de sus características de audio (`danceability`, `energy`, `valence`, `tempo`, etc.)?

- **Tipo de tarea:** clasificación multiclase.
- **Variable objetivo:** `track_genre` reagrupado a ~10 macro-géneros (del original de 125 sub-géneros).
- **Métrica primaria:** macro-F1 (balancea rendimiento entre clases).
- **Métricas secundarias:** accuracy, top-3 accuracy, matriz de confusión.

## Dataset

- **Fuente:** [maharshipandya/-spotify-tracks-dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (Kaggle) / mirror en HuggingFace.
- **Tamaño:** 114 000 tracks × 20 columnas, 125 sub-géneros originales.
- **Licencia:** BSD.
- **Aprobado por el profesor** como dataset propio fuera de la lista curada del PDF del proyecto (sección 8.3).

Los archivos de datos **no se versionan** (ver `.gitignore`). Descargar manualmente siguiendo las instrucciones de abajo y colocar en `data/raw/spotify_tracks.csv`.

## Estructura del repo

```
Proyecto_ML/
├── README.md                     este archivo
├── requirements.txt              dependencias Python
├── .gitignore
├── data/
│   ├── raw/                      datos originales (no versionados)
│   └── processed/                datos limpios intermedios (no versionados)
├── notebooks/
│   └── 01_eda_baseline.ipynb     EDA + baseline (Entrega 1)
├── src/
│   └── utils.py                  helpers de visualización y overview
├── figures/                      figuras exportadas desde notebooks
├── report/
│   ├── data_card.md              descripción del dataset
│   └── reporte_e1.md             reporte de Entrega 1 (fuente para PDF)
└── poster/                       póster (Entrega 3)
```

## Reproducibilidad

- **Python:** 3.10+
- **Semilla:** `RANDOM_STATE = 42` en todos los splits y modelos estocásticos.
- **Pipeline:** `ColumnTransformer` + `Pipeline` de `scikit-learn` para evitar leakage entre preprocesamiento y modelo.

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Descargar el dataset

Opción web:

1. Ir a https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
2. Descargar el zip y extraer.
3. Renombrar el archivo `dataset.csv` a `spotify_tracks.csv` y colocarlo en `data/raw/`.

Opción CLI (requiere cuenta Kaggle y `~/.kaggle/kaggle.json`):

```bash
kaggle datasets download -d maharshipandya/-spotify-tracks-dataset -p data/raw/ --unzip
```

### Ejecutar

```bash
jupyter lab notebooks/01_eda_baseline.ipynb
```

## Entregas

| Entrega | Peso | Fecha | Contenido |
|---|---|---|---|
| E1 | 5 % | 2026-04-26 | EDA + baseline reproducible + data card + reporte |
| E2 | 10 % | 2026-04-30 | Comparación de 2-3 familias de modelos + validación |
| E3 | 20 % | 2026-05-14 | Modelo final + interpretación (SHAP) + póster + presentación |

## Licencia

Proyecto académico. No se redistribuyen los datos originales de Spotify.
