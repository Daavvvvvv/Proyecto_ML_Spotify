# Entrega 1 — Problema, datos, EDA y baseline

**Curso:** Aprendizaje de Máquina Aplicado (ST1613/ST1631), EAFIT, 2026-1.
**Profesor:** Marco Teran.
**Equipo:** [TODO: nombres].
**Fecha de entrega:** 2026-04-26.

---

## 1. Resumen ejecutivo

[TODO: 4-6 líneas. Qué problema se resolvió, qué dataset se usó, qué métrica, resultado del baseline vs. dummy, y una frase sobre próximos pasos.]

## 2. Problema y motivación

### 2.1 Pregunta de investigación

¿Se puede predecir el **macro-género musical** de una canción a partir únicamente de sus **características de audio** (danceability, energy, valence, tempo, etc.)?

### 2.2 Motivación

[TODO: 1-2 párrafos. Por qué es interesante pedagógicamente: features con semántica clara que permiten interpretación posterior, habilita aplicaciones downstream (recomendación basada en contenido, agrupamiento por similaridad), y probar hasta qué punto el audio solo explica el género.]

### 2.3 Tipo de tarea y métrica

- **Tarea:** clasificación multiclase supervisada.
- **Variable objetivo:** `macro_genre` (≈10 clases, agrupadas desde 125 sub-géneros originales de Spotify).
- **Métrica primaria:** macro-F1. Elegida porque trata a todas las clases con igual peso y es robusta a desbalance residual tras el reagrupamiento.
- **Métricas secundarias:** accuracy, top-3 accuracy, matriz de confusión.

## 3. Datos

- **Fuente:** Spotify Tracks Dataset (`maharshipandya` en Kaggle / HuggingFace).
- **Tamaño:** 114 000 tracks × 20 columnas.
- **Licencia:** BSD.
- **Aprobación:** dataset aprobado por el profesor como propio (fuera de la lista curada del PDF, sección 8.3).

Para detalles completos ver [`data_card.md`](data_card.md).

## 4. EDA — Hallazgos principales

### 4.1 Target

[TODO: insertar `figures/target_distribution.png`. Comentar el balance después del mapeo a macro-géneros. Nota: el dataset original es balanceado por sub-género (~1000 tracks por sub-género), pero al agrupar a macro los tamaños varían — comentar cuál es la clase más y menos frecuente.]

### 4.2 Features numéricas

[TODO: insertar `figures/numerical_distributions.png` y comentar:
- Rangos observados vs. los documentados en la data card.
- Features con distribuciones bimodales o con colas largas (p.ej. `speechiness`, `instrumentalness`, `acousticness`).
- Outliers evidentes en `duration_ms`, `tempo`, `loudness`.]

### 4.3 Correlaciones

[TODO: insertar `figures/correlation_heatmap.png` y comentar:
- Correlaciones fuertes esperadas (`energy` ↔ `loudness`, `energy` ↔ `acousticness` negativa).
- Ausencia de multicolinealidad severa que obligue a remover features.]

### 4.4 Relación features–target

[TODO: insertar 2-3 boxplots por macro-género. Comentar al menos dos relaciones claras, por ejemplo:
- `acousticness` alta para folk/clásica, baja para metal/electronic.
- `danceability` alta para latin/reggaeton/electronic, media para rock, baja para clásica.]

## 5. Calidad de datos y preparación

### 5.1 Limpieza

- **Duplicados por `track_id`:** [TODO: número]. Deduplicados antes del split.
- **Missing:** [TODO: reportar si hay o no].
- **Outliers:** [TODO: criterio si se aplicó algún filtro, p.ej. descartar tracks con `duration_ms < 30s` o `tempo == 0`.]

### 5.2 Mapeo de géneros

Los 125 sub-géneros originales fueron agrupados en [TODO: N] macro-categorías. El mapeo completo está en el notebook (sección 2) y se justifica por [TODO: criterio — familia estilística, similaridad instrumental, etc.].

### 5.3 Features usadas

Se utilizan exclusivamente las **14 features de audio** listadas en la data card. Se **excluyen** intencionalmente:

- `track_id`, `track_name`, `album_name`, `artists` → son identificadores o texto libre de alta cardinalidad; incluirlos permitiría al modelo memorizar por artista/canción en vez de aprender del audio.
- `popularity` → variable inestable que cambia con el tiempo y no corresponde a una propiedad intrínseca del audio.

### 5.4 Split y pipeline

- **Split estratificado** por `macro_genre`, 80 % train / 20 % test, `random_state = 42`.
- **Pipeline `scikit-learn`:** `ColumnTransformer` (StandardScaler a numéricas, OneHotEncoder a `key`/`mode`/`time_signature` categóricos) → modelo.
- **Prevención de leakage:** el preprocesamiento se ajusta únicamente sobre el train; todas las transformaciones ocurren dentro del `Pipeline`.

## 6. Baselines y resultados

| Modelo | macro-F1 | Accuracy | Top-3 acc |
|---|---|---|---|
| Dummy (`stratified`) | [TODO] | [TODO] | [TODO] |
| Logistic Regression (multinomial) | [TODO] | [TODO] | [TODO] |

[TODO: matriz de confusión en `figures/confusion_matrix_logreg.png`. Identificar y comentar 2-3 pares de clases que se confunden más (típicamente pop↔indie, rock↔alt-rock, etc.).]

## 7. Discusión

[TODO: 1-2 párrafos.

- ¿Qué tan difícil es el problema? ¿Cuánto mejora LogReg sobre Dummy?
- ¿Qué features parecen discriminar más? (inspección de coeficientes o importancia de permutación).
- ¿Qué clases son "fáciles" y cuáles "difíciles"? Justificar con matriz de confusión.]

## 8. Limitaciones y próximos pasos

### Limitaciones

- El mapeo de géneros a macro-categorías es una decisión humana; diferentes agrupamientos pueden dar resultados distintos.
- Las features son outputs del algoritmo propietario de Spotify; no auditamos cómo se calculan.
- El dataset no es una muestra aleatoria del "universo musical": está sesgado por la API de Spotify en país, época e idioma.

### Próximos pasos (hacia Entrega 2)

- Comparar al menos 3 familias: LogReg multinomial, Random Forest, Gradient Boosting (XGBoost o LightGBM).
- Validación cruzada estratificada para estimaciones más robustas.
- Evaluación de ajuste de umbral y análisis de errores por clase.

---

**Responde también a:**

- *¿Qué problema intenta resolver?* → Sección 2.
- *¿Por qué este conjunto de datos es adecuado?* → Sección 3 + data card.
- *¿Qué métrica es razonable y por qué?* → Sección 2.3.
- *¿Cuál es el baseline y qué tan difícil parece el problema?* → Sección 6 + 7.
