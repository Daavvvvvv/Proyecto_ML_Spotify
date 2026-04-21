# Data Card — Spotify Tracks Dataset

## Identificación

- **Nombre:** Spotify Tracks Dataset
- **Autor del dataset:** `maharshipandya` (Kaggle / HuggingFace)
- **URL:** https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
- **Mirror:** https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset
- **Versión usada en este proyecto:** [TODO: fecha de descarga, hash del archivo o versión de Kaggle]
- **Licencia:** BSD.

## Procedencia

Recolectado por el autor usando la **Spotify Web API** con Python. Cada track viene con las *audio features* precalculadas internamente por Spotify.

## Tamaño y estructura

- **Filas:** 114 000 tracks.
- **Columnas:** 20.
- **Formato:** CSV, ~20 MB.
- **Granularidad:** una fila = un track (con posibilidad de duplicados si la misma canción pertenece a varios álbumes o compilaciones).

## Esquema

| Columna | Tipo | Descripción | Rango |
|---|---|---|---|
| `track_id` | str | ID único de Spotify del track | 22 chars |
| `artists` | str | Artista(s), múltiples separados por `;` | — |
| `album_name` | str | Álbum que contiene el track | — |
| `track_name` | str | Título de la canción | — |
| `popularity` | int | Popularidad 0–100 basada en plays recientes (no histórico) | 0–100 |
| `duration_ms` | int | Duración en milisegundos | 0 – 5.24M |
| `explicit` | bool | Tiene letra explícita | — |
| `danceability` | float | Qué tan bailable es | 0 – 0.99 |
| `energy` | float | Intensidad percibida | 0 – 1 |
| `key` | int | Clase de tono (0 = C, 1 = C♯/D♭, …). `-1` = no detectado | -1 – 11 |
| `loudness` | float | Volumen promedio en dB | -49.53 – 4.53 |
| `mode` | int | 1 = mayor, 0 = menor | 0 / 1 |
| `speechiness` | float | Presencia de palabra hablada | 0 – 0.97 |
| `acousticness` | float | Confianza de que sea acústica | 0 – 1 |
| `instrumentalness` | float | Probabilidad de no tener vocales | 0 – 1 |
| `liveness` | float | Probabilidad de grabación en vivo | 0 – 1 |
| `valence` | float | Positividad emocional | 0 – 1 |
| `tempo` | float | Tempo estimado en BPM | 0 – 243 |
| `time_signature` | int | Compás (3/4, 4/4, etc.) | 3 – 7 |
| `track_genre` | str | Género asignado por Spotify | 125 valores únicos |

## Variable objetivo

- **Columna original:** `track_genre` (125 sub-géneros).
- **Target usado en este proyecto:** `macro_genre`, resultado de agrupar los 125 sub-géneros en ~10 macro-categorías estilísticas (ver `notebooks/01_eda_baseline.ipynb`, sección 2).
- **Justificación del reagrupamiento:** [TODO: explicar en el reporte, típicamente por ambigüedad entre sub-géneros (`pop` vs `power-pop` vs `indie-pop`) y por mejorar la interpretabilidad de la matriz de confusión.]

## Tipo de tarea

Clasificación multiclase (supervisada).

## Limitaciones y riesgos conocidos

1. **Duplicados:** el autor advierte que el mismo track puede aparecer varias veces por formar parte de distintos álbumes o compilaciones. Se deduplica por `track_id` antes del split para evitar **leakage train/test**.
2. **Etiquetas ruidosas:** los géneros son asignados por Spotify, no por musicólogos. Hay sub-géneros ambiguos.
3. **`popularity` es inestable:** refleja tendencias actuales, no plays históricos; no se usa como target.
4. **Sesgo de muestreo:** el dataset se obtiene vía API sin garantía de cobertura proporcional por país, época o idioma. Los resultados no generalizan al "universo musical" completo.
5. **Features de audio propietarias:** `danceability`, `energy`, etc. son outputs de un algoritmo interno de Spotify no documentado en detalle. No podemos auditar cómo se calculan.

## Uso ético

- No se redistribuyen los archivos originales (solo se versiona el código y los resúmenes de EDA).
- No se identifica a usuarios — el dataset solo contiene metadatos de tracks públicos.
- No se hacen afirmaciones sobre gustos o demografía de personas reales.

## Características del split

- **Split estratificado** por `macro_genre`: 80 % train / 20 % test, `random_state = 42`.
- Deduplicación por `track_id` antes del split.
- Todas las transformaciones numéricas (`StandardScaler`) y categóricas se aprenden **solo sobre el train** y se aplican al test vía `Pipeline` de scikit-learn.
