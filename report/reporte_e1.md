# Entrega 1 — Problema, datos, EDA y baseline

**Curso:** Aprendizaje de Máquina Aplicado (ST1613/ST1631), EAFIT, 2026-1.
**Profesor:** Marco Teran.
**Equipo:** David Vélez · Daniela Villamizar · Jaymar Murillo.
**Fecha de entrega:** 2026-04-26.

---

## 1. Resumen

El presente trabajo formula un problema de clasificación supervisada multiclase orientado a predecir el macro-género musical de una canción a partir, exclusivamente, de sus características de audio extraídas mediante la API de Spotify. Tras la limpieza, el conjunto de datos final cuenta con **89 571 pistas únicas** descritas por **14 atributos de audio** (energía, *danceability*, tempo, sonoridad, *valence*, *acousticness*, entre otros), reagrupadas en **15 macro-géneros** construidos a partir de los 114 sub-géneros disponibles en el conjunto original.

La métrica primaria seleccionada es **macro-F1**, dada su robustez ante el desbalance entre clases. Los baselines obtenidos sobre el conjunto de prueba (20 % estratificado) son los siguientes:

| Modelo | macro-F1 | Accuracy | Top-3 Acc |
|---|---|---|---|
| Dummy (`stratified`) | 0.064 | 0.090 | 0.230 |
| Logistic Regression (multinomial) | **0.238** | **0.358** | **0.629** |

Logistic Regression supera al baseline aleatorio en un factor cercano a 4× en macro-F1 y 2.7× en top-3 accuracy, lo cual evidencia que las características acústicas contienen señal predictiva sobre el género. No obstante, el desempeño por clase revela limitaciones estructurales del modelo lineal: las clases `pop` e `hip-hop` obtienen F1 = 0, lo que sugiere fronteras de decisión no separables linealmente en el espacio de características empleado. Estos hallazgos motivan la exploración de familias no lineales (árboles de decisión, ensambles, *gradient boosting*) en la siguiente entrega.

---

## 2. Problema y motivación

### 2.1 Pregunta de investigación

¿Es posible predecir el macro-género musical de una canción utilizando exclusivamente sus características de audio (`danceability`, `energy`, `valence`, `tempo`, entre otras)?

### 2.2 Motivación

Las plataformas de *streaming* musical organizan y recomiendan catálogos compuestos por millones de pistas. Los sistemas tradicionales basados en filtrado colaborativo enfrentan el problema del *cold-start*, presente cuando un usuario o una pista carecen de historial suficiente para inferir afinidades. Una alternativa consiste en clasificar el contenido directamente desde sus propiedades acústicas, construyendo perfiles que no dependan del comportamiento agregado de los usuarios.

Las características de audio expuestas por la API de Spotify poseen semántica musical interpretable —tempo en BPM, sonoridad en dB, probabilidad de instrumentación acústica— lo que habilita la justificación cuantitativa de cada predicción en términos de propiedades sonoras concretas. Adicionalmente, el problema permite explorar una pregunta de carácter empírico: **¿qué proporción del género musical está determinada por la firma acústica y qué proporción depende de factores extra-sonoros (época, contexto cultural, mercado)?** Un modelo de alta precisión sustentaría que el género es fundamentalmente acústico; uno de baja precisión indicaría que existen factores externos no capturables únicamente desde el audio.

### 2.3 Tipo de tarea y métrica

- **Tarea:** clasificación multiclase supervisada.
- **Variable objetivo:** `macro_genre`, compuesta por 15 clases nominadas más una clase residual `other` (~2 %), construida mediante agrupamiento manual de los 114 sub-géneros originales (véase sección 5.2).
- **Métrica primaria:** **macro-F1**. Esta métrica calcula el promedio aritmético de los F1 por clase con peso uniforme, independientemente de su frecuencia. Penaliza modelos que ignoren clases minoritarias y premia aquellos que aprenden patrones discriminativos en todas las categorías.
- **Métricas secundarias:** *accuracy*, top-3 *accuracy* y matriz de confusión normalizada por fila.

---

## 3. Datos

- **Fuente:** Spotify Tracks Dataset, autoría de `maharshipandya` ([Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) / HuggingFace).
- **Tamaño original:** 114 000 tracks × 20 columnas (~20 MB en formato CSV).
- **Tamaño tras limpieza:** **89 571 tracks**.
- **Sub-géneros únicos:** 114.
- **Licencia:** BSD.
- **Aprobación:** el dataset fue aprobado por el profesor del curso como conjunto propio fuera de la lista curada del documento del proyecto (sección 8.3). El equipo de tres integrantes también cuenta con autorización explícita.

La descripción completa del esquema, características éticas y limitaciones se documentan en [`data_card.md`](data_card.md).

---

## 4. Análisis exploratorio

### 4.1 Distribución del objetivo

Tras agrupar los 114 sub-géneros en 15 macro-géneros más la clase residual `other`, la distribución resultante (sobre las 114 000 filas previas a la deduplicación) es la siguiente:

| macro_genre | Tracks | macro_genre | Tracks |
|---|---|---|---|
| electronic | 19 000 | ambient | 5 000 |
| rock | 15 000 | kids-comedy | 5 000 |
| latin | 12 000 | soul-funk | 5 000 |
| world | 10 000 | pop | 5 000 |
| metal | 9 000 | reggae | 4 000 |
| folk | 9 000 | classical | 3 000 |
| asian-pop | 7 000 | hip-hop | 3 000 |
| | | jazz | 1 000 |
| **other** | **2 000** | | |

Únicamente dos sub-géneros (`sad`, `romance`) permanecen sin asignar y se mantienen como clase `other`, dado su perfil acústico ambiguo (etiquetas de *mood* más que de género propiamente dicho). La clase residual representa cerca del 2 % del total. La razón entre la clase mayoritaria (`electronic`, 19 000) y la minoritaria (`jazz`, 1 000) es de 19:1, lo cual refuerza la pertinencia de macro-F1 como métrica primaria.

### 4.2 Características numéricas

Las 10 variables numéricas cubren dimensiones acústicas, rítmicas y tímbricas. Los hallazgos descriptivos más relevantes son:

- `acousticness`, `instrumentalness` y `speechiness` presentan distribuciones fuertemente sesgadas a la derecha. La mediana de `instrumentalness` es ≈ 0.000042, indicando que la mayoría de tracks contienen voz; sin embargo, la media asciende a 0.156 debido a la cola larga de pistas puramente instrumentales (clásica, ambient).
- `energy` y `danceability` exhiben distribuciones más simétricas, centradas en 0.69 y 0.58 respectivamente.
- `loudness` varía entre −49.5 dB y +4.5 dB, con mediana en −7 dB. Los valores extremos negativos se asocian a grabaciones de baja amplitud (música clásica, ambient).
- `duration_ms` presenta una cola derecha pronunciada (máximo ≈ 87 minutos). El filtrado de tracks con duración inferior a 30 s elimina la mayoría de los valores atípicos por defecto.
- `tempo` se distribuye aproximadamente normal, centrado en 122 BPM, con un patrón bimodal leve en torno a 95–100 y 120–130 BPM.

### 4.3 Correlaciones

| Par de variables | r de Pearson | Interpretación |
|---|---|---|
| `energy` ↔ `loudness` | ≈ +0.76 | Las pistas más intensas se perciben con mayor sonoridad |
| `energy` ↔ `acousticness` | ≈ −0.72 | Los instrumentos acústicos producen menor energía percibida |
| `valence` ↔ `danceability` | ≈ +0.40 | Las composiciones con mayor positividad emocional tienden a ser más bailables |

No se observa multicolinealidad severa que obligue a eliminar variables. El nivel de correlación es manejable mediante la regularización L2 incorporada en Logistic Regression y resulta irrelevante para los modelos basados en árboles que se evaluarán en la Entrega 2.

### 4.4 Relación entre características y objetivo

Los perfiles medianos de las características clave por macro-género permiten observar que las clases ocupan regiones distinguibles del espacio acústico:

| macro_genre | energy | acousticness | danceability | valence |
|---|---|---|---|---|
| metal | 0.890 | 0.003 | 0.472 | 0.401 |
| latin | 0.744 | 0.126 | 0.740 | 0.657 |
| reggae | 0.736 | 0.127 | 0.756 | 0.679 |
| asian-pop (`j-pop`) | 0.723 | 0.150 | 0.551 | 0.549 |
| electronic | 0.718 | 0.063 | 0.664 | 0.362 |
| rock | 0.703 | 0.096 | 0.552 | 0.546 |
| hip-hop | 0.690 | 0.124 | 0.753 | 0.550 |
| pop | 0.618 | 0.281 | 0.642 | 0.498 |
| folk | 0.550 | 0.456 | 0.564 | 0.478 |
| world | 0.550 | 0.126 | 0.424 | 0.209 |
| soul-funk | 0.526 | 0.350 | 0.609 | 0.513 |
| jazz | 0.332 | 0.787 | 0.499 | 0.504 |
| ambient | 0.178 | 0.921 | 0.364 | 0.125 |
| classical | 0.142 | 0.973 | 0.377 | 0.346 |

Las observaciones principales son:

- `energy` y `acousticness` constituyen las variables más discriminativas a nivel global. Los géneros metal, classical y ambient ocupan extremos opuestos del espacio acústico: metal concentra los valores más altos de energía con `acousticness` cercana a cero, mientras que classical y ambient presentan energías inferiores a 0.2 con `acousticness` superiores a 0.9.
- `danceability` separa los géneros rítmicamente bailables (latin, reggae, hip-hop, con valores medianos en torno a 0.74–0.76) de aquellos con estructuras rítmicas más libres (rock, pop, en torno a 0.55–0.64).
- `valence` revela polarización emocional: latin y reggae alcanzan los valores más altos (~0.66–0.68); metal, world y ambient los más bajos.
- La clase pop se ubica acústicamente en posición central: ninguna de sus medianas alcanza valores extremos, lo cual permite anticipar dificultades de discriminación para un clasificador lineal.

---

## 5. Calidad de datos y preparación

### 5.1 Limpieza

| Paso | Detalle | Filas resultantes |
|---|---|---|
| Carga inicial | — | 114 000 |
| Deduplicación por `track_id` | Las mismas pistas aparecen en múltiples álbumes y compilaciones (advertencia documentada por el autor) | 89 741 (−24 259) |
| Filtrado de outliers | `duration_ms < 30 000 ms` o `tempo == 0` | **89 571** (−170) |

Se identificaron tres filas con valores faltantes en las columnas `artists`, `album_name` y `track_name`. Dado que estas columnas no se utilizan como variables predictoras, no se aplicó imputación ni eliminación.

### 5.2 Mapeo de géneros

Los 114 sub-géneros únicos fueron agrupados en 15 macro-categorías más una clase residual `other` de tamaño marginal:

| macro_genre | Sub-géneros incluidos |
|---|---|
| **rock** | rock, rock-n-roll, hard-rock, alt-rock, alternative, grunge, psych-rock, punk, punk-rock, rockabilly, emo, garage, indie, j-rock, goth |
| **metal** | metal, heavy-metal, black-metal, death-metal, grindcore, metalcore, hardcore, industrial, happy |
| **pop** | pop, power-pop, indie-pop, synth-pop, pop-film |
| **asian-pop** | j-pop, k-pop, cantopop, mandopop, j-idol, j-dance, anime |
| **electronic** | electronic, edm, house, deep-house, techno, trance, dubstep, drum-and-bass, minimal-techno, idm, electro, progressive-house, chicago-house, detroit-techno, breakbeat, hardstyle, club, dance, party |
| **hip-hop** | hip-hop, r-n-b, trip-hop |
| **latin** | latin, latino, reggaeton, salsa, samba, tango, brazil, pagode, forro, mpb, sertanejo, spanish |
| **jazz** | jazz |
| **classical** | classical, opera, piano |
| **folk** | folk, country, bluegrass, honky-tonk, acoustic, singer-songwriter, songwriter, blues, guitar |
| **reggae** | reggae, dub, ska, dancehall |
| **ambient** | ambient, chill, sleep, study, new-age |
| **world** | afrobeat, indian, iranian, turkish, malay, british, french, german, swedish, world-music |
| **kids-comedy** | children, kids, disney, comedy, show-tunes |
| **soul-funk** | soul, funk, gospel, groove, disco |
| **other** | sad, romance |

### 5.2.1 Metodología del agrupamiento

El reagrupamiento se construyó aplicando tres criterios secuenciales, alineados con la pregunta de investigación:

**Criterio 1: Similitud acústica (precedencia primaria).** Dado que el modelo solo observa las 14 variables de audio —sin acceso al idioma, marketing o país—, los sub-géneros con firma acústica indistinguible se agrupan en la misma macro-categoría aunque presenten diferencias culturales. Las reglas operativas, expresadas sobre las medianas observadas, fueron las siguientes:

| Regla observacional (medianas) | Macro-género asignado |
|---|---|
| `energy` > 0.85 ∧ `acousticness` < 0.05 | metal |
| `energy` ∈ [0.6, 0.85] ∧ `acousticness` < 0.15 ∧ `danceability` < 0.6 | rock |
| `danceability` > 0.65 ∧ `valence` > 0.55 | pop / latin |
| `instrumentalness` > 0.3 ∧ `acousticness` > 0.4 | classical / ambient |
| `danceability` > 0.65 ∧ `instrumentalness` > 0.05 | electronic |

**Criterio 2: Consistencia interna.** Para cada macro-género se documenta la regla aplicada, evitando que una misma lógica produzca decisiones contradictorias. Por ejemplo, `ska` se asigna a reggae —en lugar de a rock-punk— por su tempo sincopado y `danceability` ≈ 0.72.

**Criterio 3: Verificación empírica de fronteras.** Los sub-géneros cuya asignación resultaba ambigua entre dos macro-categorías (`industrial`, `hardcore`, `happy`, `goth`, `j-rock`) se resolvieron consultando las medianas reales del conjunto, en lugar de basarse en convenciones musicológicas.

**Decisiones notables:**

- **`industrial` → metal.** Mediana de `energy` = 0.92 con `acousticness` ≈ 0; perfil indistinguible del de `heavy-metal`. Su asignación a rock habría incrementado la varianza intra-clase de rock sin ganancia discriminativa.
- **`hardcore` → metal.** Las medianas (0.898 / 0.008) se aproximan al núcleo de metal.
- **`happy` → metal.** Pese a la denotación nominal, su perfil (`energy` 0.944, `valence` 0.27, tempo 160 BPM) se corresponde con el sub-género *happy hardcore*, caracterizado por tempos extremos y agresividad acústica equiparable a metal.
- **`alternative`, `garage`, `emo`, `goth` → rock.** Confirmado empíricamente: las medianas se ubican dentro del rango operativo de rock.
- **`party` → electronic.** Energía 0.91, *danceability* 0.67 y *valence* 0.71 configuran un perfil EDM/club indistinguible de `dance`.
- **`j-rock` → rock; `j-pop`, `k-pop`, `cantopop`, `mandopop`, `j-idol` → asian-pop.** Esta aparente inconsistencia se sustenta así: los pops asiáticos comparten convenciones de producción comercial específicas (estilo *idol*, sintetizadores característicos) y concentran un volumen suficiente para sostener una clase propia. `j-rock`, en cambio, presenta firma acústica idéntica al rock occidental; separarlo introduciría una clase con escasos ejemplos cuyo perfil colapsaría con rock. Se acepta el compromiso: la matriz de confusión esperada mostrará a `asian-pop` parcialmente confundida con `pop`, hallazgo consistente con la limitación intrínseca de las características de audio para distinguir origen geográfico.
- **`children`, `kids`, `disney`, `comedy`, `show-tunes` → kids-comedy.** Aunque su `danceability` mediana es alta (~0.71), su `acousticness` (~0.55) y `energy` (~0.43) difieren del perfil de pop comercial. Asignar estos sub-géneros a pop habría incrementado la varianza intra-clase de pop.
- **`sad`, `romance` → other.** Constituyen etiquetas de *mood* o contexto con perfiles heterogéneos: `sad` combina alta `danceability` con baja `valence`; `romance` presenta `acousticness` extrema (0.95) acompañada de voz (`instrumentalness` ≈ 0). Su retención en `other` evita la contaminación de las macro-clases bien definidas.

### 5.3 Variables predictoras

Se utilizan exclusivamente las 14 características de audio descritas en la *data card*. Se excluyen de manera intencional las siguientes columnas:

- `track_id`, `track_name`, `album_name`, `artists`: identificadores y texto libre de alta cardinalidad cuya inclusión permitiría al modelo memorizar combinaciones de artista o canción, en lugar de aprender patrones acústicos.
- `popularity`: variable inestable que cambia con el tiempo y no refleja una propiedad intrínseca del audio.

### 5.4 Partición y *pipeline*

- **Partición estratificada** por `macro_genre`: 80 % entrenamiento (71 656 tracks) / 20 % prueba (17 915 tracks), con `random_state = 42`.
- ***Pipeline* de scikit-learn**: `ColumnTransformer` aplicando `StandardScaler` a las 10 variables numéricas y `OneHotEncoder(handle_unknown='ignore')` a las 4 categóricas (`key`, `mode`, `time_signature`, `explicit`), seguido del modelo.
- **Prevención de fugas de información**: las transformaciones se ajustan exclusivamente sobre el conjunto de entrenamiento; toda manipulación de variables ocurre dentro del `Pipeline`, garantizando que el conjunto de prueba permanece aislado durante el ajuste.

---

## 6. Baselines y resultados

### 6.1 Resultados globales

| Modelo | macro-F1 | Accuracy | Top-3 acc |
|---|---|---|---|
| Dummy (`stratified`) | 0.064 | 0.090 | 0.230 |
| Logistic Regression (multinomial) | **0.238** | **0.358** | **0.629** |

Logistic Regression supera al Dummy estratificado en un factor cercano a 4× en macro-F1, 4× en *accuracy* y 2.7× en top-3 *accuracy*. La consistencia de la mejora a través de las tres métricas indica que el modelo aprende señal predictiva real desde las características de audio, en lugar de explotar la distribución a priori de las clases.

### 6.2 Resultados por clase (Logistic Regression)

| Clase | Precision | Recall | F1 | Soporte |
|---|---|---|---|---|
| ambient | 0.418 | 0.473 | **0.444** | 960 |
| asian-pop | 0.217 | 0.062 | 0.097 | 1 254 |
| classical | 0.373 | 0.181 | 0.244 | 502 |
| electronic | 0.426 | 0.706 | **0.531** | 2 978 |
| folk | 0.273 | 0.391 | 0.322 | 1 423 |
| hip-hop | 0.000 | 0.000 | 0.000 | 447 |
| jazz | 0.304 | 0.067 | 0.109 | 105 |
| kids-comedy | 0.432 | 0.304 | 0.357 | 969 |
| latin | 0.333 | 0.490 | 0.397 | 1 835 |
| metal | 0.523 | 0.609 | **0.563** | 1 485 |
| other | 0.254 | 0.174 | 0.206 | 293 |
| pop | 0.000 | 0.000 | 0.000 | 712 |
| reggae | 0.268 | 0.038 | 0.067 | 575 |
| rock | 0.265 | 0.396 | 0.317 | 1 932 |
| soul-funk | 0.174 | 0.006 | 0.011 | 690 |
| world | 0.199 | 0.103 | 0.136 | 1 755 |

Las tres clases con mejor desempeño (metal F1 = 0.563, electronic F1 = 0.531, ambient F1 = 0.444) corresponden a categorías cuyos perfiles acústicos identificados en la sección 4.4 ocupan extremos del espacio de características: metal en los valores altos de `energy`, electronic en la combinación intermedia de `instrumentalness` y `danceability`, ambient en los valores extremos de `acousticness` e `instrumentalness`.

Las clases pop e hip-hop registran F1 = 0, lo cual indica que el modelo lineal no genera predicciones positivas para ellas. Sus perfiles medianos las sitúan en regiones centrales del espacio de características —pop con valores promedio en todas las dimensiones; hip-hop con `danceability` 0.75 (similar a latin y reggae) y `valence` 0.55 (similar a rock)—. Las fronteras de decisión que las separarían de sus clases vecinas no son linealmente separables con las variables empleadas.

---

## 7. Discusión

### 7.1 Dificultad del problema

Con 16 clases efectivas y un macro-F1 de Dummy igual a 0.064, el desempeño aleatorio se sitúa próximo al 6 %. Logistic Regression alcanza 0.238 —cerca de cuatro veces el azar— pero permanece distante de los niveles requeridos para sistemas en producción (típicamente F1 > 0.5 en tareas comparables). El problema presenta dificultad moderada y deja margen amplio para modelos más expresivos.

### 7.2 Patrones del modelo lineal

- **Clases con buena separabilidad lineal (F1 > 0.40):** `metal`, `electronic`, `ambient`. Comparten perfiles acústicos extremos en al menos una variable —metal con `energy` ≈ 0.89, electronic con la combinación de `danceability` e `instrumentalness`, ambient con `acousticness` ≈ 0.92—. Estas clases son linealmente separables con las características actuales.
- **Clases con separabilidad parcial (F1 entre 0.20 y 0.40):** `latin`, `kids-comedy`, `folk`, `rock`, `classical`. Cada una posee al menos una variable distintiva, pero comparten regiones del espacio con clases vecinas (rock con metal, folk con jazz, classical con ambient).
- **Clases sin separación lineal efectiva (F1 ≈ 0):** `pop` e `hip-hop`. Sus perfiles medianos se ubican en el centro del espacio. El modelo lineal no logra construir fronteras de decisión que las distingan de sus clases vecinas (rock, latin, electronic), lo que provoca que las predicciones se asignen sistemáticamente a clases más extremas.

### 7.3 Análisis del caso `asian-pop`

Como se anticipó al definir el mapeo (sección 5.2.1), `asian-pop` colapsa contra `pop` y otras categorías comerciales (F1 = 0.097). Este resultado valida empíricamente la hipótesis planteada: la separación entre pop asiático y pop occidental es defendible desde la perspectiva cultural, pero acústicamente las clases resultan indistinguibles para un clasificador lineal. El hallazgo es metodológicamente útil: las características de audio de Spotify capturan las dimensiones musicales pero no las dimensiones culturales o lingüísticas.

### 7.4 Implicaciones para la Entrega 2

El análisis sugiere las siguientes líneas de trabajo:

- **Familias no lineales**: la incorporación de Random Forest, Gradient Boosting (XGBoost, LightGBM) o SVM con núcleo no lineal puede capturar interacciones entre variables que resulten críticas para distinguir pop e hip-hop de sus clases vecinas.
- **Manejo del desbalance**: el uso de `class_weight='balanced'` o estrategias de muestreo estratificado debería incrementar la representación de las clases minoritarias durante el entrenamiento.
- **Evaluación de la categoría `asian-pop`**: si los modelos no lineales tampoco logran separarla, será evidencia robusta de que la distinción es extra-acústica y debería evaluarse su absorción dentro de `pop`.

---

## 8. Limitaciones y trabajo futuro

### Limitaciones reconocidas

- El mapeo a 15 macro-categorías constituye una decisión humana sustentada por evidencia empírica, pero no auditada por especialistas en musicología. La existencia de categorías como `asian-pop` y `kids-comedy` representa compromisos pragmáticos derivados del análisis de los datos.
- Las características de audio son resultados del algoritmo propietario de Spotify; no se cuenta con documentación detallada sobre su cálculo interno.
- El conjunto no constituye una muestra aleatoria del universo musical: existe sesgo hacia producción mayoritaria, occidental y del periodo 2000–2020.
- La clase residual `other` (~2 %) introduce ruido marginal por contener etiquetas de *mood* (`sad`, `romance`) en lugar de géneros propiamente musicales.

### Trabajo futuro

1. **Comparación de al menos tres familias de modelos**: Logistic Regression multinomial (baseline actual), Random Forest y Gradient Boosting (XGBoost o LightGBM).
2. **Validación cruzada estratificada** de cinco particiones, con el fin de obtener estimaciones más estables que las del *holdout* simple.
3. **Análisis de errores por clase**, con énfasis en `pop` e `hip-hop`, dada su falta de separabilidad lineal.
4. **Importancia de variables**: combinación de coeficientes de Logistic Regression e *importancia por permutación* para confirmar las características que mejor discriminan cada macro-género.
5. **Ajuste de pesos por clase** para equilibrar el aprendizaje en las categorías minoritarias.
