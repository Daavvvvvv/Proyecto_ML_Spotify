# Entrega 2 — Modelado, validación honesta y comparación estadística

**Curso:** Aprendizaje de Máquina Aplicado (ST1613/ST1631), EAFIT, 2026-1.
**Profesor:** Marco Teran.
**Equipo:** David Vélez · Daniela Villamizar · Jaymar Murillo.
**Fecha de entrega:** 2026-04-30.

---

## 1. Resumen

La presente entrega construye sobre los baselines de E1 mediante (i) la corrección de una **fuga de datos por artista** detectada durante la auditoría inicial, (ii) la implementación de un esquema de validación cruzada agrupado y estratificado, y (iii) la comparación estadística rigurosa de tres familias de modelos.

Los hallazgos centrales son los siguientes:

1. **Detección y corrección de leakage.** El partido aleatorio empleado en E1 permitía que canciones del mismo artista cayeran simultáneamente en entrenamiento y prueba, lo cual sesgaba al modelo a reconocer la firma sonora del artista en lugar del género. Se reemplazó por `StratifiedGroupKFold` agrupado por `artists`, garantizando solapamiento nulo entre particiones.
2. **Comparación de tres familias** (`LogisticRegression`, `RandomForestClassifier`, `HistGradientBoostingClassifier`) mediante validación cruzada de cinco particiones sobre el 80 % de entrenamiento, con un *holdout* del 20 % intocado hasta la evaluación final.
3. **Tests estadísticos pareados** (t-test relacionado, Wilcoxon signed-rank, tamaño del efecto Cohen's *d*) con corrección de Bonferroni para tres comparaciones. Los modelos basados en árboles superan al lineal de forma altamente significativa (*p* < 0.001 post-corrección, |*d*| > 10); entre `RandomForestClassifier` y `HistGradientBoostingClassifier` la diferencia no es estadísticamente significativa (*p* = 0.18 post-corrección).
4. **Modelo seleccionado para E3:** `HistGradientBoostingClassifier` por mayor *macro-F1* medio en CV (0.328 vs 0.320 de Random Forest). La evaluación final sobre el *holdout* arroja macro-F1 = 0.339, accuracy = 0.433, balanced accuracy = 0.332 y top-3 accuracy = 0.692.

---

## 2. Corrección metodológica (errata de E1)

### 2.1 Diagnóstico

Durante el inicio de E2 se efectuó una auditoría sistemática del flujo de validación de E1, en línea con el criterio de "rigor experimental y validación honesta" (20 % del peso evaluativo, sección 7 del documento del proyecto). El diagnóstico identificó **una fuga de datos crítica**:

| # | Riesgo | Severidad | Estado en E1 |
|---|---|---|---|
| 1 | Leakage por artista en el *split* aleatorio | **Crítico** | Presente |
| 2 | Estimación basada en un único *split* sin CV | Medio | Presente |
| 3 | Mapeo de géneros informado por estadísticas del *dataset* completo | Bajo (decisión consciente) | Documentado |
| 4 | Uso de `popularity`, `track_name`, `artists` como *features* | — | Ya excluidos en E1 |
| 5 | Transformaciones aprendidas fuera del `Pipeline` | — | Ya manejado en E1 |
| 6 | Duplicados por `track_id` | — | Ya deduplicados en E1 |

### 2.2 Evidencia cuantitativa del leakage

El conjunto contiene **31 386 artistas únicos** distribuidos en 89 571 *tracks*, con un promedio de 2.85 *tracks* por artista. **11 265 artistas (35.9 %) poseen dos o más canciones**, concentrando aproximadamente el 70 % de las filas. Los artistas con mayor cantidad de pistas son:

| Artista | Tracks |
|---|---|
| George Jones | 260 |
| my little airport | 171 |
| The Beatles | 149 |
| BTS | 143 |
| Håkan Hellström | 141 |

Bajo un *split* aleatorio 80/20, cada uno de estos artistas tendría con altísima probabilidad ~80 % de sus canciones en entrenamiento y ~20 % en prueba. El modelo no aprende a generalizar a artistas nuevos: aprende la firma acústica de artistas conocidos y la aplica al conjunto de prueba. Esta condición infla artificialmente las métricas reportadas en E1.

### 2.3 Corrección aplicada

La sustitución del *split* aleatorio por `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, tomando el primer *fold* como conjunto de prueba (~20 %), garantiza simultáneamente:

- **Estratificación** por `macro_genre` (preservando proporciones de las 16 clases en *train* y *test*).
- **Agrupamiento** por `artists`: ningún artista aparece en ambos conjuntos.

Verificación post-corrección sobre los datos limpios:

- Train: 71 657 filas, 25 100 artistas únicos.
- Test: 17 914 filas, 6 286 artistas únicos.
- **Solapamiento de artistas entre train y test: 0.**

El *notebook* `01_eda_baseline.ipynb` fue re-ejecutado con la corrección aplicada y se incorporó una celda *markdown* de "Errata metodológica" al inicio. El documento `reporte_e1.pdf` se mantiene inalterado como registro de la entrega original.

### 2.4 Impacto sobre las métricas reportadas en E1

| Modelo | macro-F1 (E1 leaky) | macro-F1 (E1 corregido) | Δ |
|---|---|---|---|
| Dummy (`stratified`) | 0.064 | 0.063 | −0.001 |
| Logistic Regression | **0.238** | **0.223** | **−0.015** |

El efecto del leakage sobre Logistic Regression es modesto (≈1.5 pts F1) porque el modelo lineal no logra explotar la firma acústica de cada artista. La magnitud del sesgo sería sustancialmente mayor en familias no lineales (árboles, *boosting*), las cuales sí pueden memorizar patrones específicos de artista en su capacidad. Por esta razón la comparación entre familias presentada en esta entrega requería forzosamente el esquema de validación agrupado.

---

## 3. Estrategia de validación sin leakage

### 3.1 Esquema general

| Conjunto | Tamaño | Uso |
|---|---|---|
| **Train (CV interno)** | 71 657 (80 %) | `StratifiedGroupKFold(n_splits=5)` para comparar modelos y obtener distribución de *scores* |
| **Test (holdout)** | 17 914 (20 %) | Evaluación única del modelo seleccionado al final de E2 |

### 3.2 CV interno

Sobre el conjunto de entrenamiento se aplicó `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, generando 5 particiones validación/entrenamiento. Verificación de ausencia de leakage en cada *fold*:

| Fold | Tamaño train | Tamaño val | Artistas val | Solapamiento |
|---|---|---|---|---|
| 1 | 57 337 | 14 320 | 5 031 | 0 |
| 2 | 57 307 | 14 350 | 5 002 | 0 |
| 3 | 57 337 | 14 320 | 5 027 | 0 |
| 4 | 57 335 | 14 322 | 5 039 | 0 |
| 5 | 57 312 | 14 345 | 5 001 | 0 |

### 3.3 Métricas evaluadas

Para cada modelo se capturaron cuatro métricas por *fold*:

- **`macro_f1` (métrica primaria)** — promedio de F1 por clase con peso uniforme. Robusta al desbalance.
- **`accuracy`** — proporción global de aciertos.
- **`balanced_accuracy`** — promedio del *recall* por clase.
- **`top3_accuracy`** — la clase real está entre las tres más probables; relevante para clases acústicamente vecinas.

---

## 4. Familias de modelos comparadas

Tres familias representativas del espectro de modelos supervisados de tabla:

| Modelo | Familia | Hiperparámetros |
|---|---|---|
| `LogisticRegression` | Lineal | `multi_class='multinomial'`, `max_iter=2000`, `C=1.0` por defecto |
| `RandomForestClassifier` | Ensemble — bagging | `n_estimators=200`, `max_depth=None`, `n_jobs=-1` |
| `HistGradientBoostingClassifier` | Ensemble — boosting | `max_iter=200`, *early stopping* automático, defaults restantes |

`HistGradientBoostingClassifier` fue preferido sobre XGBoost/LightGBM dado que (i) está incluido en *scikit-learn* (no introduce dependencias externas), (ii) ofrece desempeño comparable a las alternativas en problemas tabulares densos, y (iii) preserva la coherencia de `requirements.txt` heredada de E1.

Las tres familias se encapsularon en `Pipeline` con el `ColumnTransformer` (StandardScaler para variables numéricas, OneHotEncoder para categóricas) definido en E1.

---

## 5. Resultados de validación cruzada

### 5.1 Tabla principal

Media ± desviación estándar sobre los cinco *folds* del CV interno (`StratifiedGroupKFold`, agrupado por `artists`):

| Modelo | macro-F1 | accuracy | balanced acc. | top-3 acc. |
|---|---|---|---|---|
| `logreg` | 0.234 ± 0.004 | 0.361 ± 0.008 | 0.250 ± 0.005 | 0.628 ± 0.010 |
| `random_forest` | 0.320 ± 0.005 | **0.432 ± 0.006** | 0.325 ± 0.005 | **0.692 ± 0.008** |
| `hist_gb` | **0.328 ± 0.007** | 0.426 ± 0.007 | **0.327 ± 0.007** | 0.690 ± 0.008 |

### 5.2 Intervalos de confianza al 95 %

Construidos por *bootstrap* con 10 000 remuestreos sobre los cinco *scores* por modelo (métrica primaria macro-F1):

| Modelo | media | CI inferior 95 % | CI superior 95 % |
|---|---|---|---|
| `logreg` | 0.2342 | 0.2299 | 0.2372 |
| `random_forest` | 0.3200 | 0.3154 | 0.3241 |
| `hist_gb` | **0.3276** | 0.3214 | 0.3327 |

El intervalo de `logreg` está completamente separado de los intervalos de los modelos basados en árboles. Los intervalos de `random_forest` y `hist_gb` **se solapan** en el rango [0.3214, 0.3241], anticipando que la diferencia entre ambos puede no ser estadísticamente significativa.

---

## 6. Comparación estadística entre modelos

### 6.1 Tests pareados sobre macro-F1

Para cada par de modelos se aplicaron dos tests pareados sobre los cinco *scores* por *fold*: t-test relacionado (paramétrico) y Wilcoxon signed-rank (no paramétrico). Se reporta el tamaño del efecto Cohen's *d* para muestras pareadas (`d = media(diff) / std(diff)`). Los *p*-values fueron corregidos por Bonferroni multiplicando por el número de comparaciones (*n* = 3).

| A | B | media(A) − media(B) | t-stat | *p* (t, Bonf) | Cohen's *d* | Sig. (t)? |
|---|---|---|---|---|---|---|
| `logreg` | `random_forest` | −0.0859 | −22.87 | 0.0001 | −10.23 | **Sí** |
| `logreg` | `hist_gb` | −0.0934 | −24.44 | 0.0001 | −10.93 | **Sí** |
| `random_forest` | `hist_gb` | −0.0075 | −2.59 | 0.181 | −1.16 | No |

### 6.2 Interpretación

- **`logreg` vs ensembles**: la diferencia es altamente significativa (*p* < 0.001 post-Bonferroni) con tamaños de efecto extraordinariamente grandes (|*d*| > 10, donde la convención considera *d* = 0.8 como un efecto "grande"). El salto al ensemble corresponde a una mejora de ≈ 9 pts F1 medios.
- **`random_forest` vs `hist_gb`**: la diferencia (0.0075 en macro-F1) no alcanza significancia estadística al nivel α = 0.05 corregido por Bonferroni. El tamaño del efecto (|*d*| ≈ 1.2) sí es notable, pero con cinco observaciones por modelo la prueba carece de potencia suficiente para concluir.
- **Wilcoxon**: con n = 5 *folds*, el *p*-value mínimo alcanzable por Wilcoxon es 0.0625 (bilateral), por lo que ningún test no paramétrico puede ser significativo a α = 0.05 incluso con orden perfecto entre los pares. Este es un límite conocido del test para muestras pequeñas y no contradice los resultados del t-test, sino que ilustra una restricción del diseño con cinco *folds*.

### 6.3 Selección del modelo ganador

`hist_gb` exhibe el mayor *macro-F1* medio, pero su superioridad sobre `random_forest` no es estadísticamente significativa. Bajo este escenario el criterio metodológico admite dos opciones:

1. **Seleccionar por media**: `hist_gb`.
2. **Seleccionar por simplicidad/costo**: `random_forest` (≈3× más rápido en entrenamiento; 22 s vs 72 s sobre el train completo).

Para esta entrega se aplica el criterio (1) por consistencia con la métrica primaria definida en E1. La discusión sobre eventual reconsideración para E3 se incluye en la sección 9.

---

## 7. Evaluación final en holdout

El modelo seleccionado (`hist_gb`) se re-entrenó sobre el 80 % de entrenamiento completo y se evaluó **una única vez** sobre el *holdout* de 17 914 filas, intocado durante todo el proceso de selección y comparación.

### 7.1 Métricas globales

| Métrica | Test holdout | CV mean | Δ (test − CV) |
|---|---|---|---|
| macro-F1 | 0.339 | 0.328 | +0.012 |
| accuracy | 0.433 | 0.426 | +0.007 |
| balanced accuracy | 0.332 | 0.327 | +0.005 |
| top-3 accuracy | 0.692 | 0.690 | +0.001 |

El delta positivo y pequeño respecto a la media de CV (+0.012 en macro-F1) indica que el modelo generaliza al menos tan bien sobre el *holdout* como sobre los *folds* internos, descartando sobreajuste a los conjuntos de validación cruzada.

### 7.2 Resultados por clase

| Clase | Precision | Recall | F1 | Soporte |
|---|---|---|---|---|
| electronic | 0.547 | 0.715 | **0.620** | 2 978 |
| metal | 0.604 | 0.603 | **0.604** | 1 485 |
| ambient | 0.512 | 0.555 | **0.532** | 960 |
| kids-comedy | 0.539 | 0.469 | 0.501 | 969 |
| latin | 0.428 | 0.536 | 0.476 | 1 835 |
| folk | 0.382 | 0.444 | 0.411 | 1 423 |
| classical | 0.477 | 0.349 | 0.403 | 501 |
| asian-pop | 0.371 | 0.381 | 0.376 | 1 254 |
| rock | 0.304 | 0.371 | 0.334 | 1 932 |
| world | 0.305 | 0.290 | 0.297 | 1 755 |
| other | 0.308 | 0.150 | 0.201 | 294 |
| jazz | 0.571 | 0.117 | 0.194 | 103 |
| reggae | 0.244 | 0.118 | 0.159 | 575 |
| soul-funk | 0.262 | 0.086 | 0.129 | 690 |
| pop | 0.219 | 0.074 | 0.111 | 713 |
| hip-hop | 0.195 | 0.049 | 0.079 | 447 |

**Patrones observados:**

- Las cinco clases con mejor F1 (electronic, metal, ambient, kids-comedy, latin) coinciden con las categorías que en el EDA de E1 ocupaban regiones acústicas distinguibles del espacio de *features*.
- Las clases que en E1 obtuvieron F1 ≈ 0 (`pop`, `hip-hop`) muestran mejora moderada con el modelo no lineal (0.111 y 0.079 respectivamente), confirmando la hipótesis de E1 de que el problema requería fronteras no lineales. La mejora absoluta sigue siendo limitada, lo que sugiere que la separabilidad acústica de estas clases con respecto a `rock` / `latin` / `electronic` es estructuralmente baja.
- `jazz` presenta el mayor desbalance precision/recall (0.571 / 0.117): el modelo es preciso cuando predice jazz, pero subdetecta la clase. Con solo 103 ejemplos en el *holdout* (~0.6 %), el modelo prioriza clases mayoritarias.

---

## 8. Discusión

### 8.1 Validación honesta como criterio metodológico

La corrección del *leakage* por artista representa la contribución metodológica central de E2. El descubrimiento confirma que un *split* aparentemente correcto (estratificado por clase) puede contener fugas si la estructura del conjunto incluye agrupamientos implícitos —en este caso, múltiples canciones por artista comparten firma acústica—. La aplicación de `StratifiedGroupKFold` constituye el patrón apropiado en *datasets* con esta característica y debería extenderse a cualquier estudio musical basado en *features* de Spotify.

El procedimiento aplicado, **detección — documentación — corrección — re-ejecución**, prioriza la transparencia metodológica sobre la apariencia de un avance monótono entre entregas. La caída marginal de macro-F1 de Logistic Regression (de 0.238 a 0.223) es consecuencia esperable y deseable de una validación más estricta.

### 8.2 Magnitud del salto entre familias

La diferencia entre el modelo lineal y los ensembles es uniforme y dramática: ≈9 pts F1 (efecto |*d*| > 10). Esto valida la hipótesis del informe de E1 (sección 7.4) de que el problema requería modelos capaces de capturar interacciones no lineales entre las *features* de audio. En particular:

- El *recall* de `electronic` pasa de 0.706 (LogReg E1) a 0.715 (HGB E2). El modelo lineal ya hacía buen trabajo en la clase mayoritaria.
- El F1 de `pop` pasa de 0.000 a 0.111 y el de `hip-hop` de 0.000 a 0.079. La capacidad no lineal sí permite construir fronteras de decisión en regiones centrales del espacio, aunque la separabilidad estructural de estas clases sigue siendo limitada.

### 8.3 Empate estadístico entre Random Forest e HistGradientBoosting

La ausencia de significancia entre los dos ensembles no es un resultado adverso: indica que ambos son aproximadamente equivalentes para este problema con los hiperparámetros razonables empleados. Para E3 esto abre dos líneas de exploración:

1. **Tuning sistemático** mediante `HalvingRandomSearchCV` agrupado por artista, sobre el espacio de hiperparámetros de cada familia. Es posible que el ranking cambie con búsquedas exhaustivas.
2. **Stacking o promedio de probabilidades** entre `random_forest` y `hist_gb`. Dado que las dos familias presentan errores parcialmente decorrelacionados (diferentes mecanismos de partición del espacio), un *ensemble* podría superar a cualquiera individualmente.

### 8.4 Brecha entre desempeño actual y referencia

El macro-F1 de 0.339 sobre 16 clases efectivas supera al *baseline* de 0.064 en un factor de 5.3×. Sigue siendo distante del rango típico para sistemas en producción (F1 > 0.5), reforzando que el problema posee dificultad estructural: las *features* de audio capturan adecuadamente géneros con perfiles acústicos extremos pero discriminan mal entre categorías comerciales centrales (pop, rock, hip-hop) que comparten regiones del espacio.

---

## 9. Limitaciones y trabajo futuro (E3)

### Limitaciones reconocidas en E2

- **Cinco *folds* limitan el poder estadístico del test pareado.** Una alternativa sería `RepeatedStratifiedGroupKFold` con 10 repeticiones x 5 *folds*, pero el costo computacional sería de aproximadamente 10× respecto al actual.
- **No se aplicó tuning de hiperparámetros** dentro de E2. Los valores empleados son razonables pero no óptimos. Esto se aborda explícitamente en E3.
- **No se exploró rebalanceo de clases** (`class_weight`, *SMOTE*, *undersampling*). El rebalanceo debería evaluarse con cuidado bajo el esquema agrupado: el muestreo sintético podría introducir leakage si no se aplica dentro de cada *fold*.
- **`hist_gb` fue seleccionado por mayor media en CV** pese a no haber diferencia significativa con `random_forest`. La elección quedaría sujeta a revisión si el *tuning* o el *stacking* de E3 modifican el orden.

### Plan de trabajo para E3

1. **Búsqueda de hiperparámetros agrupada por artista** sobre `random_forest` y `hist_gb` con `HalvingRandomSearchCV` y `StratifiedGroupKFold` interno.
2. **Stacking de probabilidades** entre los dos modelos no lineales, evaluando si el meta-clasificador supera a sus base-models.
3. **Interpretación con SHAP** del modelo final, identificando: (i) las *features* de audio más discriminativas por clase, (ii) las interacciones que los modelos lineales no capturaron, (iii) las regiones del espacio donde se concentran las confusiones recurrentes (rock↔pop, latin↔reggae, electronic↔house).
4. **Evaluación cualitativa de `asian-pop`**: si el modelo final no logra separarla de `pop`, se reconsiderará su unificación con `pop` siguiendo el principio de la hipótesis acústica formulada en E1.
5. **Póster y presentación** consolidando hallazgos de las tres entregas.

---

## 10. Reproducibilidad

- *Notebook* principal: `notebooks/02_modeling_e2.ipynb`.
- *Notebook* corregido de E1: `notebooks/01_eda_baseline.ipynb` (con errata documentada).
- Mapeo de géneros centralizado: `src/genre_mapping.py`.
- *Scores* por *fold* persistidos en: `data/processed/cv_scores.csv` (60 filas: 3 modelos × 4 métricas × 5 *folds*).
- Figuras generadas: `figures/cv_macro_f1_by_model.png`, `figures/cv_macro_f1_with_ci.png`, `figures/confusion_matrix_hist_gb_holdout.png`.
- Semilla aleatoria: `RANDOM_STATE = 42` consistente entre E1 y E2.
- Dependencias declaradas en `requirements.txt` (sin adiciones respecto a E1).
