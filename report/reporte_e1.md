# Entrega 1 — Problema, datos, EDA y baseline

**Curso:** Aprendizaje de Máquina Aplicado (ST1613/ST1631), EAFIT, 2026-1.
**Profesor:** Marco Teran.
**Equipo:** David Vélez - Daniela Villamizar - Jaymar Murillo.
**Fecha de entrega:** 2026-04-26.

---

## 1. Resumen ejecutivo

Este proyecto aborda la clasificación supervisada multiclase del macro-género musical de una canción utilizando exclusivamente sus características de audio extraídas por la API de Spotify. El conjunto de datos utilizado incluye 89,571 pistas únicas con 14 características acústicas (entre las que se encuentran la energía, la danza, el tempo, la sonoridad, la valencia y la acústica), organizadas en 11 géneros principales. La métrica principal es la macro-F1, que fue seleccionada por su fortaleza frente al desequilibrio de clases. El baseline ingenuo (DummyClassifier estratificado) alcanzó un accuracy del 31.0% y una puntuación de macro-F1 de 0.093, lo que confirma que el problema no es trivial y que hay mucho espacio para mejorar. Se empleó la regresión logística multinomial como línea base real; no fue posible ejecutar localmente debido a un problema de compatibilidad con scikit-learn ≥ 1.6, aunque se ha aplicado la corrección.

---

## 2. Problema y motivación

### 2.1 Pregunta de investigación

¿Se puede predecir el **macro-género musical** de una canción a partir únicamente de sus **características de audio** (danceability, energy, valence, tempo, etc.)?

### 2.2 Motivación

Las plataformas de streaming como Spotify deben organizar y recomendar millones de canciones. Los sistemas de recomendación se equivocan en el inicio en frío, cuando un usuario o una pista es nuevo y no hay datos colaborativos a la mano. Una opción es clasificar el audio según sus características acústicas de manera directa, elaborando perfiles de contenido que no dependan del historial.

Las características de audio de Spotify contienen una semántica musical directa, como el tempo en BPM, la sonoridad en dB y la probabilidad de instrumento acústico en términos de acústica. Esto permite que los resultados puedan ser interpretados, es decir, que se pueda justificar una clasificación en términos de sonidos reales. Asimismo, el problema posibilita la exploración de un límite científico: ¿qué proporción del género es determinada por el sonido y qué proporción por factores como la época, el contexto cultural o la mercadotecnia? Un modelo de precisión alta indicaría que el género es fundamentalmente acústico; uno de baja precisión señalaría que existen factores externos que no están incluidos en las características.

### 2.3 Tipo de tarea y métrica

- **Tarea:** clasificación multiclase supervisada.
- **Variable objetivo:** `macro_genre` (11 clases, agrupadas desde 114 sub-géneros originales de Spotify).
- **Métrica primaria:** macro-F1. Elegida porque promedia el F1 de cada clase con igual peso, independientemente de su frecuencia. Un modelo que siempre prediga `other` (49 % del dataset) alcanzaría ≈ 49 % de accuracy sin aprender nada sobre los demás géneros; macro-F1 lo penalizaría directamente.
- **Métricas secundarias:** accuracy, top-3 accuracy, matriz de confusión normalizada por fila.

---

## 3. Datos

- **Fuente:** Spotify Tracks Dataset — `maharshipandya` en Kaggle / HuggingFace.
- **Tamaño original:** 114,000 tracks × 20 columnas. ~20 MB CSV.
- **Tamaño tras limpieza:** 89,571 tracks.
- **Licencia:** BSD.
- **Aprobación:** dataset aprobado por el profesor como propio (fuera de la lista curada del PDF, sección 8.3).

---

## 4. EDA — Hallazgos principales

### 4.1 Target

Tras mapear 114 sub-géneros a 11 macro-géneros, la distribución resultante muestra desbalance notable:

| macro_genre | Tracks | macro_genre | Tracks |
|---|---|---|---|
| other | 56,000 | metal | 5,000 |
| electronic | 14,000 | folk | 4,000 |
| latin | 11,000 | reggae | 4,000 |
| rock | 8,000 | classical | 3,000 |
| pop | 6,000 | hip-hop | 2,000 |
| | | jazz | 1,000 |

`other` concentra el 49 % del dataset porque 56 de los 114 sub-géneros (afrobeat, ambient, anime, soul, funk, etc.) no han sido asignados a ninguna macro-categoría todavía. Para las clases nombradas, la ratio entre la más frecuente (electronic, 14 k) y la menos frecuente (jazz, 1 k) es 14:1. El dataset original tenía ≈ 1,000 tracks por sub-género; al agrupar a macro, los tamaños difieren según cuántos sub-géneros caen en cada familia.


### 4.2 Features numéricas

Los 10 features numéricos cubren dimensiones acústicas, rítmicas y tímbricas:

- **`acousticness`, `instrumentalness`, `speechiness`** presentan distribuciones fuertemente sesgadas a la derecha. La mediana de `instrumentalness` es ≈ 0.000042 (casi todos los tracks tienen voz), pero la media sube a 0.156 por la cola larga de tracks puramente instrumentales.
- **`energy`** y **`danceability`** son más simétricas, centradas en 0.64 y 0.57 respectivamente.
- **`loudness`** varía de −49.5 dB a +4.5 dB con mediana en −7 dB; los valores extremos negativos corresponden a grabaciones muy silenciosas (clásica, ambient).
- **`duration_ms`** tiene cola derecha pronunciada (máximo ≈ 87 minutos). Tras filtrar tracks < 30 s, el rango queda en valores razonables para canciones y álbumes en vivo.
- **`tempo`** es aproximadamente normal centrado en 122 BPM con un leve patrón bimodal en ≈ 95–100 y 120–130 BPM.

### 4.3 Correlaciones

| Par de features | r de Pearson | Interpretación |
|---|---|---|
| `energy` ↔ `loudness` | ≈ +0.76 | Tracks más intensas percibidas como más fuertes |
| `energy` ↔ `acousticness` | ≈ −0.72 | Instrumentos acústicos son menos energéticos |
| `valence` ↔ `danceability` | ≈ +0.40 | Canciones alegres tienden a ser más bailables |

No hay multicolinealidad severa que obligue a eliminar features. El nivel de correlación observado es manejable con regularización L2 estándar en Regresión Logística y no afecta los modelos de árbol planificados para la Entrega 2.


### 4.4 Relación features–target

Los boxplots por macro-género muestran que al menos cuatro features son discriminativas:

- **`acousticness`:** es la feature más discriminativa. Classical y folk tienen mediana > 0.70; metal, electronic y rock tienen mediana cercana a 0.
- **`energy`:** metal presenta mediana ≈ 0.90 (máxima intensidad); classical < 0.20 (mínima). Electronic y rock se ubican en 0.70–0.85.
- **`danceability`:** latin y reggae lideran con medianas ≈ 0.70–0.75; classical y metal quedan por debajo de 0.40.
- **`valence`:** latin y reggae tienen valencia alta (canciones positivas); metal y classical tienen valencia baja o neutral.

La clase `other` muestra dispersión amplia en todas las features, confirmando su naturaleza heterogénea y la necesidad de completar el mapeo.

---

## 5. Calidad de datos y preparación

### 5.1 Limpieza

| Paso | Detalle | Filas resultantes |
|---|---|---|
| Carga inicial | — | 114,000 |
| Deduplicación por `track_id` | Misma canción en varios álbumes o compilaciones | 89,741 (−24,259) |
| Filtrado de outliers | `duration_ms < 30,000 ms` ó `tempo == 0` | **89,571** (−170) |

Valores faltantes: 3 filas en `artists`, `album_name` y `track_name`. Corresponden únicamente a columnas de metadatos que no se usan como features, por lo que no se imputan ni eliminan.

### 5.2 Mapeo de géneros

Los 114 sub-géneros únicos fueron agrupados en **11 macro-categorías** según familia estilística e instrumentación:

| macro_genre | Sub-géneros incluidos (selección) |
|---|---|
| rock | rock, alt-rock, hard-rock, punk, punk-rock, grunge, indie, rock-n-roll |
| metal | metal, black-metal, death-metal, heavy-metal, metalcore |
| pop | pop, power-pop, indie-pop, j-pop, k-pop, pop-film |
| electronic | electronic, edm, house, deep-house, techno, trance, dubstep, drum-and-bass, minimal-techno, idm, electro, progressive-house, chicago-house, detroit-techno |
| hip-hop | hip-hop, r-n-b |
| latin | latin, latino, reggaeton, salsa, samba, tango, brazil, pagode, forro, mpb, sertanejo, spanish |
| jazz | jazz |
| classical | classical, opera, piano |
| folk | folk, country, bluegrass, honky-tonk |
| reggae | reggae, dub, ska, dancehall |
| other | 56 sub-géneros sin asignar (afrobeat, ambient, anime, chill, soul, funk, etc.) |

El criterio de agrupamiento combina: (1) similitud de instrumentación (acústica vs. eléctrica vs. electrónica), (2) origen cultural para géneros latinoamericanos, y (3) estructura rítmica predominante. Completar el mapeo de `other` es el principal trabajo pendiente.

### 5.3 Features usadas

Se utilizan exclusivamente las **14 features de audio** listadas en la data card. Se **excluyen** intencionalmente:

- `track_id`, `track_name`, `album_name`, `artists` → son identificadores o texto libre de alta cardinalidad; incluirlos permitiría al modelo memorizar por artista/canción en vez de aprender del audio.
- `popularity` → variable inestable que cambia con el tiempo y no corresponde a una propiedad intrínseca del audio.

### 5.4 Split y pipeline

- **Split estratificado** por `macro_genre`, 80 % train (71,656) / 20 % test (17,915), `random_state = 42`.
- **Pipeline `scikit-learn`:** `ColumnTransformer` con `StandardScaler` para las 10 features numéricas y `OneHotEncoder(handle_unknown='ignore')` para las 4 categóricas (`key`, `mode`, `time_signature`, `explicit`) → modelo.
- **Prevención de leakage:** el preprocesamiento se ajusta únicamente sobre el train; todas las transformaciones ocurren dentro del `Pipeline`.

---

## 6. Baselines y resultados

| Modelo | macro-F1 | Accuracy | Top-3 acc |
|---|---|---|---|
| Dummy (`stratified`) | **0.093** | **0.310** | **0.395** |
| Logistic Regression (multinomial) | pendiente* | pendiente* | pendiente* |

*El notebook original incluía el argumento `multi_class="multinomial"` que fue eliminado en scikit-learn ≥ 1.6. La corrección (remover dicho parámetro, que ahora es el comportamiento por defecto) está aplicada en el notebook. Los resultados numéricos exactos de LogReg están pendientes de re-ejecución; la figura `figures/confusion_matrix_logreg.png` fue generada en el entorno de un integrante del equipo con scikit-learn 1.4.x.

---

## 7. Discusión

El macro-F1 del DummyClassifier (0.093) confirma que el problema no es trivialmente fácil: con 11 clases desbalanceadas el azar proporcional apenas alcanza el 9 % en macro-F1. El EDA anticipa que la dificultad varía significativamente entre clases:

- **Clases probablemente fáciles:** `classical` y `metal` tienen perfiles acústicos casi opuestos en `acousticness` y `energy`; incluso un modelo lineal debería separarlos con F1 elevado. `latin` y `reggae` se distinguen por `danceability` y `valence` notoriamente superiores al resto.
- **Clases probablemente difíciles:** `pop`, `rock` e `indie` comparten rangos similares en casi todas las features. La confusión entre estas clases es intrínseca y probablemente requiere modelos no lineales para reducirse.
- **Obstáculo principal:** la clase `other` agrupa 56 sub-géneros musicalmente heterogéneos. Hasta que se complete el mapeo, cualquier modelo verá ruido de etiqueta severo en el 49 % de los datos, lo que deprimirá el macro-F1 global independientemente de la capacidad del algoritmo.

Las features con mayor poder discriminativo esperado, según el EDA, son `acousticness`, `energy`, `danceability`, `speechiness` e `instrumentalness`. La inspección de coeficientes de la Regresión Logística y el análisis de importancia de permutación se realizarán una vez re-ejecutado el notebook con la corrección aplicada.

---

## 8. Limitaciones y próximos pasos

### Limitaciones

- El mapeo de géneros a macro-categorías es una decisión humana subjetiva; 56 sub-géneros permanecen en `other` en esta entrega.
- Las features son outputs del algoritmo propietario de Spotify; no auditamos cómo se calculan internamente.
- El dataset no es una muestra aleatoria del "universo musical": está sesgado por la cobertura de la API de Spotify (sesgo hacia mainstream, occidente, años 2000–2020).
- La clase `other` introduce ruido de etiqueta severo al concentrar géneros musicalmente incompatibles.
