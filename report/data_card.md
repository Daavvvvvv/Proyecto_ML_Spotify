# Data Card — Spotify Tracks Dataset

## 1. Identificación

- **Nombre:** Spotify Tracks Dataset.
- **Autor:** `maharshipandya` (Kaggle / HuggingFace).
- **URL principal:** https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
- **Mirror en HuggingFace:** https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset
- **Versión utilizada:** descarga del 2026-04-26 desde Kaggle.
- **Licencia:** BSD.

## 2. Procedencia

El conjunto fue construido por el autor original mediante consultas a la **Spotify Web API** utilizando Python. Cada registro incluye las *audio features* precalculadas internamente por Spotify para la pista correspondiente.

## 3. Tamaño y estructura

- **Filas (CSV original):** 114 000 tracks.
- **Filas tras limpieza:** **89 571 tracks** (deduplicación por `track_id` y filtrado de outliers básicos).
- **Columnas:** 20.
- **Formato:** CSV, ~20 MB.
- **Granularidad:** una fila representa un track. El conjunto original presenta duplicados, dado que la misma pista puede aparecer en múltiples álbumes o compilaciones.
- **Sub-géneros únicos:** 114.

## 4. Esquema

| Columna | Tipo | Descripción | Rango |
|---|---|---|---|
| `track_id` | str | Identificador único de Spotify | 22 caracteres |
| `artists` | str | Artista o artistas, separados por `;` | — |
| `album_name` | str | Álbum que contiene el track | — |
| `track_name` | str | Título de la canción | — |
| `popularity` | int | Popularidad 0–100 basada en plays recientes (no histórico) | 0 – 100 |
| `duration_ms` | int | Duración en milisegundos | 0 – 5.24 M |
| `explicit` | bool | Indica si la letra es explícita | — |
| `danceability` | float | Idoneidad para baile (estabilidad de tempo, regularidad rítmica) | 0 – 0.99 |
| `energy` | float | Intensidad percibida | 0 – 1 |
| `key` | int | Clase de tono (0 = C, 1 = C♯/D♭, …); −1 indica sin detección | −1 – 11 |
| `loudness` | float | Sonoridad promedio en dB | −49.53 – 4.53 |
| `mode` | int | Modalidad: 1 = mayor, 0 = menor | 0 / 1 |
| `speechiness` | float | Presencia de palabra hablada | 0 – 0.97 |
| `acousticness` | float | Confianza de naturaleza acústica | 0 – 1 |
| `instrumentalness` | float | Probabilidad de ausencia de voces | 0 – 1 |
| `liveness` | float | Probabilidad de grabación en vivo | 0 – 1 |
| `valence` | float | Positividad emocional | 0 – 1 |
| `tempo` | float | Tempo estimado en BPM | 0 – 243 |
| `time_signature` | int | Compás (3/4, 4/4, etc.) | 3 – 7 |
| `track_genre` | str | Género asignado por Spotify | 114 valores únicos |

## 5. Variable objetivo

- **Columna original:** `track_genre` (114 sub-géneros).
- **Variable construida:** `macro_genre`, resultado de agrupar los 114 sub-géneros en **15 macro-categorías** (rock, metal, pop, asian-pop, electronic, hip-hop, latin, jazz, classical, folk, reggae, ambient, world, kids-comedy, soul-funk), más una clase residual `other` (~2 % del conjunto) que recoge los sub-géneros `sad` y `romance`.
- **Justificación del agrupamiento:** combinación de tres criterios secuenciales —similitud de firma acústica (medianas de `energy`, `acousticness`, `danceability`, entre otras), consistencia interna del mapeo, y verificación empírica de fronteras a partir de los datos del propio CSV. La metodología completa se documenta en `reporte_e1.md`, sección 5.2.1.

## 6. Tipo de tarea

Clasificación multiclase supervisada (16 clases efectivas: 15 macro-géneros más la clase `other`).

## 7. Limitaciones y riesgos conocidos

1. **Duplicados.** El autor advierte que un mismo track puede aparecer múltiples veces por pertenencia a distintos álbumes o compilaciones. Se aplica deduplicación por `track_id` antes de la partición train/test, evitando fuga de información entre conjuntos.
2. **Etiquetas con ruido semántico.** Los géneros son asignados por Spotify, no por especialistas en musicología. Algunas etiquetas describen *mood* o contexto de uso (`happy`, `sad`, `romance`) más que género propiamente dicho.
3. **Inestabilidad de `popularity`.** Esta variable refleja tendencias actuales basadas en plays recientes, no plays acumulados. No se utiliza como objetivo ni como predictor.
4. **Sesgo de muestreo.** El conjunto se obtiene mediante consultas a la API de Spotify sin garantía de cobertura proporcional por país, época o idioma. Existe sesgo hacia producción mayoritaria, occidental y del periodo 2000–2020. Los resultados no son extrapolables al universo musical completo.
5. **Variables propietarias.** `danceability`, `energy`, etc. son resultados del algoritmo interno de Spotify, sin documentación pública detallada sobre su cálculo.

## 8. Consideraciones éticas

- No se redistribuyen los archivos originales: el repositorio del proyecto incluye únicamente el código de procesamiento y los resúmenes derivados del análisis.
- No se identifica a usuarios; el conjunto contiene exclusivamente metadatos de tracks publicados.
- No se realizan inferencias ni afirmaciones sobre gustos, preferencias o demografía de personas reales.

## 9. Características de la partición (Entrega 1)

- **Partición estratificada** por `macro_genre`: 80 % entrenamiento (71 656 tracks) / 20 % prueba (17 915 tracks), con `random_state = 42`.
- Deduplicación por `track_id` aplicada antes de la partición.
- Las transformaciones (`StandardScaler` para variables numéricas, `OneHotEncoder` para categóricas) se ajustan exclusivamente sobre el conjunto de entrenamiento y se aplican al conjunto de prueba mediante el `Pipeline` de scikit-learn, evitando fugas de información.
