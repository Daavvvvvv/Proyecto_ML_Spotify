# Entrega 3 — Modelo final, confiabilidad e interpretación

**Curso:** Aprendizaje de Máquina Aplicado (ST1613/ST1631), EAFIT, 2026-1.
**Equipo:** David Vélez · Daniela Villamizar · Jaymar Murillo.

---

## ¿Cuál es el mejor modelo y por qué?

### Progresión a lo largo del proyecto

| Entrega | Modelo | macro-F1 holdout | Mejora sobre anterior |
|---|---|---|---|
| E1 | DummyClassifier (stratified) | 0.063 | — |
| E1 | LogisticRegression | 0.223 | +0.160 |
| E2 | HistGradientBoostingClassifier | **0.339** | +0.116 |
| E3 | RF/HGB tuned + Stacking (pendiente) | — | esperado +0.01–0.02 |

### Modelo provisional confirmado: `HistGradientBoostingClassifier` (E2)

Seleccionado por mayor macro-F1 medio en validación cruzada agrupada (0.328 ± 0.007 sobre 5 folds `StratifiedGroupKFold` por artista). Evaluado una única vez sobre el holdout de 17,914 filas:

| Métrica | Holdout E2 | CV E2 | Δ |
|---|---|---|---|
| macro-F1 | **0.339** | 0.328 | +0.012 |
| accuracy | 0.433 | 0.426 | +0.007 |
| balanced accuracy | 0.332 | 0.327 | +0.005 |
| top-3 accuracy | **0.692** | 0.690 | +0.001 |

El delta positivo y pequeño descarta sobreajuste. El modelo generaliza al menos tan bien sobre datos no vistos como sobre los folds internos.

### Metodología de selección en E3

El notebook implementa el siguiente criterio formal para la selección final:

```
si stacking es estadísticamente significativo (t-test Bonferroni, α=0.05)
   Y la ganancia sobre la mejor base tuneada > 0.005 macro-F1:
     → modelo final = stacking
si no:
     → modelo final = mejor base tuneada (rf_tuned o hgb_tuned)
```

**Por qué esta familia de modelos es la mejor:**
Los ensembles de árboles superan al modelo lineal en ~9 pts de macro-F1 (|Cohen's d| > 10, p < 0.001 post-Bonferroni en E2). La razón es estructural: el espacio de features de audio contiene interacciones no lineales que la Regresión Logística no puede capturar. `energy` × `acousticness` separa mejor que cada feature por separado; `danceability` × `tempo` distingue latin de reggae de forma no lineal. Los árboles de decisión capturan estas interacciones naturalmente en cada nodo de partición.

---

## ¿Qué tan confiables son los resultados?

### 1. Validación sin leakage

El leakage por artista detectado en E1 fue corregido en E2. Solapamiento de artistas entre train y test: **0** en todos los conjuntos.

| Conjunto | Filas | Artistas únicos | Solapamiento |
|---|---|---|---|
| Train | 71,657 | 25,100 | 0 |
| Test (holdout) | 17,914 | 6,286 | 0 |

Sin esta corrección, LogReg mostraba macro-F1 = 0.238 (inflado +0.015 por el leakage). Todos los resultados de E2 y E3 usan el esquema agrupado.

### 2. Validación cruzada con 5 folds agrupados

`StratifiedGroupKFold(n_splits=5)` sobre el 80% de train. Ningún artista compartido entre fold de entrenamiento y fold de validación en ninguna de las 5 particiones. Desviación estándar de macro-F1 entre folds: ±0.007 para HGB, indicando estimaciones estables.

### 3. Bootstrap CI sobre el holdout

El notebook ejecuta 10,000 remuestreos bootstrap del holdout para construir el intervalo de confianza al 95% del macro-F1 final. Este intervalo cuantifica la incertidumbre de estimación sobre el conjunto de prueba específico.

### 4. Tuning con `HalvingRandomSearchCV`

La búsqueda de hiperparámetros usa el mismo `StratifiedGroupKFold` agrupado por artista como CV interno, garantizando que el proceso de tuning no filtra información de artistas del conjunto de validación. El espacio de búsqueda cubre:

- RF: `n_estimators` ∈ {200, 400, 800}, `max_depth` ∈ {None, 16, 24}, `min_samples_leaf` ∈ {1, 4, 16}, `max_features` ∈ {sqrt, 0.5, 0.7}, `class_weight` ∈ {None, balanced, balanced_subsample} → 243 configuraciones posibles.
- HGB: `learning_rate` ∈ {0.05, 0.1, 0.2}, `max_iter` ∈ {200, 400, 800}, `max_leaf_nodes` ∈ {31, 63, 127}, `min_samples_leaf` ∈ {20, 50, 100}, `l2_regularization` ∈ {0.0, 0.5, 1.0} → 243 configuraciones posibles.

### 5. Evaluación del holdout una única vez

El holdout de 17,914 filas se mantuvo intocado durante todo el proceso de E2 (comparación de modelos, selección) y E3 (tuning, stacking, selección del modelo final). Se evalúa una sola vez sobre el modelo seleccionado, sin posibilidad de sobreajuste al conjunto de prueba.

---

## ¿Qué variables o patrones explican el desempeño?

### Features más discriminativas

El análisis del EDA de E1 identificó las features con mayor separación entre clases. Los resultados por clase de E2 confirman cuáles clases se benefician de cuáles features:

| Feature | Efecto observado | Clases que separa |
|---|---|---|
| `acousticness` | Feature más discriminativa del dataset | classical/folk/ambient (alta) vs metal/electronic/rock (baja) |
| `energy` | Segundo discriminador más potente | metal (≈0.89) vs classical (≈0.15) |
| `danceability` | Separa géneros rítmicos de no rítmicos | latin/reggae (≈0.74) vs classical/metal (≈0.40) |
| `speechiness` | Identifica contenido vocal hablado | hip-hop (alta) vs instrumental (baja) |
| `instrumentalness` | Distingue piezas sin voz | classical/jazz/ambient (alta) vs pop/rock (≈0) |
| `valence` | Positividad emocional | latin/reggae (alta) vs metal/classical (baja) |

### Clases con mejor F1 — por qué son separables

| Clase | F1 (E2) | Explicación acústica |
|---|---|---|
| electronic | 0.620 | Energy alta + acousticness ≈ 0 + danceability media. Perfil único. |
| metal | 0.604 | Energy máxima + acousticness mínima + valence baja. Extremo del espacio. |
| ambient | 0.532 | Acousticness alta + energy baja + instrumentalness alta. Opuesto a metal. |
| kids-comedy | 0.501 | Speechiness alta + valence alta + tempo moderado. Región propia. |
| latin | 0.476 | Danceability + valence altas. Separable de reggae por tempo y speechiness. |

### Clases con peor F1 — por qué son difíciles

| Clase | F1 (E2) | Explicación del problema |
|---|---|---|
| hip-hop | 0.079 | Recall 0.049. Comparte región acústica con pop y latin; el factor distintivo es cultural/lingüístico, no acústico. |
| pop | 0.111 | Precision 0.219. Ocupa el centro del espacio de features; no tiene perfil acústico extremo en ninguna dimensión. |
| soul-funk | 0.129 | Recall 0.086. Acústicamente intermedio entre hip-hop, jazz y pop. La distinción es histórica/cultural. |
| reggae | 0.159 | Confundido con latin (mismo rango de danceability y valence). El tempo y el riddim son similares. |

### Análisis SHAP

El notebook calcula:
- **Importancia global SHAP**: `mean(|SHAP|)` promediado sobre 1,000 muestras del test y todas las clases. Permite confirmar si `acousticness` y `energy` dominan consistentemente sobre todas las clases.
- **Top-3 features por clase**: para cada macro-género, las 3 variables con mayor `mean(|SHAP|)` en esa clase específica.
- **Permutation importance**: verificación independiente de SHAP usando degradación del macro-F1 al permutar cada feature.

### Ablación asian-pop

El notebook compara macro-F1 con 16 clases vs. fusionando asian-pop → pop. Si la diferencia es negativa (fusionar baja el macro-F1), las dos clases son acústicamente distinguibles y deben mantenerse separadas. Si es positiva o nula, la distinción es cultural y no es capturable solo con audio features.

---

## ¿Qué conclusiones útiles deja el proyecto?

### 1. Las audio features de Spotify capturan el género acústico, no el cultural

Los géneros con perfil acústico extremo o único (electronic, metal, ambient, classical) alcanzan F1 ≥ 0.40–0.62. Los géneros definidos principalmente por contexto cultural, idioma o mercado (pop, hip-hop, soul-funk, reggae) quedan por debajo de F1 = 0.16. Esto confirma la hipótesis original: **las audio features capturan aproximadamente la mitad del problema del género musical** — el resto está fuera del alcance de este dataset.

### 2. El leakage por artista es el principal sesgo en datasets musicales

Un split aleatorio sobre un dataset con múltiples canciones por artista contamina cualquier evaluación. La corrección con `StratifiedGroupKFold` agrupado por artista es el patrón correcto para cualquier estudio musical basado en features de Spotify o similares. El impacto en este proyecto fue moderado para LogReg (+0.015 macro-F1) pero hubiera sido mayor para modelos no lineales con más capacidad de memorización.

### 3. Los modelos no lineales son necesarios pero insuficientes

El salto de LogReg a HGB fue de +0.116 macro-F1 (factor 1.5×). Sin embargo, el techo observado (~0.34 macro-F1 con 16 clases) está determinado por la separabilidad acústica inherente del problema, no por la capacidad del modelo. El tuning y el stacking de E3 se esperan que contribuyan +0.01–0.02 adicionales — mejoras marginales respecto al salto E1→E2.

### 4. El top-3 accuracy es la métrica más útil para producción

Con top-3 accuracy = 0.692, el modelo acierta el género en sus tres predicciones más probables el 69% de las veces. En un sistema de recomendación real, presentar las 3 probabilidades más altas como "géneros candidatos" es mucho más útil que forzar una predicción única de argmax.

### 5. La clase `other` confirma que el mapeo de géneros es un problema abierto

La residual `other` (~2% del dataset, solo `sad` y `romance`) muestra F1 = 0.201, aceptable para su tamaño. Sin embargo, los 6 sub-géneros que permanecen clasificables pero difíciles (pop, hip-hop, soul-funk, reggae, world, asian-pop) concentran la mayor parte de los errores del modelo y representan el trabajo de mapeo más complejo.

---

## ¿Qué haría falta para mejorar o desplegar la solución?

### Para mejorar el desempeño

| Mejora | Impacto esperado | Complejidad |
|---|---|---|
| Features no acústicas (idioma de la letra, país, año de lanzamiento) | Alto — resolver la separabilidad cultural | Media-alta |
| Re-mapeo de géneros con musicólogos | Alto — reducir ruido de etiqueta | Alta |
| `class_weight='balanced'` o SMOTE dentro del fold | Medio — mejorar recall de hip-hop/jazz/pop | Baja |
| Ajuste de umbral por clase (threshold calibration) | Medio — mejorar F1 de clases pequeñas como jazz | Baja |
| Más datos de clases difíciles (data augmentation de audio) | Alto — mejorar pop/hip-hop | Alta |
| `RepeatedStratifiedGroupKFold` (5×10) | Bajo — más poder estadístico | Media |

### Para desplegar la solución

| Requisito | Descripción |
|---|---|
| **Calibración de probabilidades** | Platt scaling o isotonic regression sobre un holdout separado. Las probabilidades raw de HGB no están calibradas y el modelo devuelve predicciones duras. |
| **Versionado de datos** | DVC (Data Version Control) para reproducibilidad bit-a-bit. El dataset CSV no está versionado en git — cualquier re-descarga de Kaggle podría introducir diferencias. |
| **API de inferencia** | Endpoint REST (FastAPI) que recibe las 14 features de audio y devuelve top-3 géneros con probabilidades. |
| **Monitoreo de drift** | Las features de Spotify pueden cambiar su semántica con actualizaciones del API. Un monitor de distribución sobre las predicciones del modelo detectaría cambios silenciosos. |
| **Pipeline de re-entrenamiento** | El dataset de Spotify tiene sesgo temporal (dominado por 2000–2020). Para mantener vigencia, el modelo debería re-entrenarse periódicamente con tracks recientes. |
| **Salida probabilística (top-3)** | En lugar de un solo género, devolver los 3 géneros con mayor probabilidad. Top-3 accuracy = 0.692 (E2), vs accuracy argmax = 0.433. |

### Limitación estructural irreducible

Parte del error residual del modelo (~66% de los casos en macro-F1 space) es **no resoluble solo con audio features**. Los géneros pop, hip-hop, soul-funk y reggae están definidos por contexto cultural, idioma, mercado y épocas históricas que no se reflejan en las 14 variables acústicas de Spotify. Para resolver estos casos se requieren features de distinta naturaleza (texto, metadatos, contexto geográfico), lo que escapa al alcance del dataset empleado.