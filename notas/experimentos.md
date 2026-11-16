# Bitácora de experimentos · Kaggle AP 2026 Q2

Competencia: predecir si un ítem de MercadoLibre es **nuevo o usado**. Métrica: **accuracy**.
Datos: 70.000 ítems de train (53,7% nuevos, 46,3% usados) y 30.000 de test, en JSON lines con muchos campos anidados.

Este documento registra **todas** las pruebas, en orden, hayan servido o no. Cada experimento tiene su notebook ejecutada en `notebooks/NN_nombre.ipynb` (autocontenida: corre sola con la carpeta de datos al lado).

## Resumen

| # | Experimento | Accuracy CV (5 folds) | Público Kaggle | ¿Sirvió? |
|---|---|---|---|---|
| 01 | Baseline: regresión logística, 8 features | 0.8177 ± 0.0039 | – | Punto de referencia |
| 02 | LightGBM, 65 features tabulares | 0.9069 ± 0.0022 | – | Sí (+0.089) |
| 03 | + modelo de texto y target encoding (meta-features OOF) | 0.9378 ± 0.0022 | 0.93971 (sub01) | Sí (+0.031) |
| 04 | + 57 features nuevas en 5 grupos | 0.9424 ± 0.0014 | 0.94304 (sub02) | Sí (+0.0046) |
| 05 | LightGBM con hiperparámetros de Optuna | 0.9432 ± 0.0017 | – | Poco (+0.0008) |
| 06 | XGBoost / CatBoost | 0.9401 / 0.9392 | – | Solos no; sirven para ensamblar |
| 07 | Ensamble LGBM + LGBM tuneado + CatBoost | 0.9442 | 0.94495 (sub03) | Sí (+0.0010) |
| 08 | Predicciones de los compañeros de vendedor | 0.9399 | – | **No** (−0.0025) |
| 09 | Modelo de texto v2 (NB-SVM, más campos) + keywords del análisis de errores | 0.9428 (config EXP04) / 0.9440 (tuneado) | – | Poco (+0.0004 / +0.0008) |
| 10 | CatBoost con texto v2 + keywords, y ensambles con modelos v3 | Cat 0.9394; mejor combinación 0.9449 | – | Dudoso: diferencias dentro del ruido |
| 11 | Promedio de semillas (5 LGBM, 3 Cat) + umbral; ensamble 1:1:1 | 0.9447 | 0.94576 (sub04) | Sí, mejora estable (+0.0005 vs EXP07) |
| 12 | kNN de títulos (proporción de usados entre los títulos más parecidos de train) | 0.9431 / 0.9439 (tuneado) | – | No (+0.0003 / −0.0001) |
| 13 | Pseudo-labeling con predicciones seguras de test | 0.9433–0.9437 vs 0.9440 | – | **No** (empeora) |
| 14 | Target encoding de interacciones del vendedor (× tipo de publicación, categoría, cantidad = 1, precio) | 0.9432 / **0.9446** (tuneado) | – | Poco (+0.0004 / +0.0006) |
| 15 | Modelos v4 (v3 + TE de interacciones) con semillas, ensamble 1:1:1 | **0.9451** | **0.94671 (sub05)** | Sí (+0.0004, mejora en 4 de 5 folds) |
| 16 | Modelo lineal grande (logística sobre todo el texto + one-hot de todo) | 0.9290 | – | Solo bien, pero no suma al ensamble |
| 18 | CatBoost con el título como texto nativo y variantes de profundidad | 0.9401 → **0.9429** (depth 10) | – | Mejora CatBoost solo, pero no suma al ensamble |
| 19 | Precio relativo a los títulos más parecidos (vecinos de train + test) | 0.9446 vs 0.9446 | – | No |
| 17 | Re-tuneo de LightGBM con Optuna sobre v4 (50 trials) | 0.9449 vs 0.9446 (1 semilla) | – | Ver EXP20 |
| 20 | Parámetros de EXP17 con 5 semillas y en el ensamble | 0.9448 vs 0.9447; ensamble 0.9449 vs 0.9451 | – | **No** (la mejora era ruido) |
| 21 | Validación adversarial: ¿se distingue train de test? | AUC 0.498 | – | Diagnóstico: no hay corrimiento, la CV representa bien a test |
| 22 | Dato externo: embeddings de títulos con un modelo de lenguaje preentrenado (MiniLM multilingüe) | sola 0.7249; LGBM 0.9444 / 0.9443 vs 0.9447 | – | **No** |
| 23 | Features v5: geometría de fotos, título editado, brecha de stock, fechas | 0.9450 vs 0.9447 (LGBM tuneado, 3 semillas) | – | Poco (+0.0003; lo que suma es la brecha de stock) |
| 24 | TE de "firmas" del vendedor: garantía exacta, tipo de fotos, envío, medios de pago, primera palabra | 0.9452 | – | Poco (+0.0002) |
| 25 | 1000 palabras del título como columnas del LightGBM | **0.9458** (3000 palabras: 0.9457) | – | Sí (+0.0006, mejora en 5 de 5 folds) |
| 26 | Modelos v7 (= v4 + EXP23–25) con semillas, ensamble 1:1:1 (CatBoost con título nativo) | **0.9462** | 0.94657 (sub06) | Sí (+0.0011 vs sub05, 4 de 5 folds) |
| 27 | Vecino más parecido dentro del mismo vendedor | 0.9457 vs 0.9458 | – | No |
| 28 | Más texto crudo: palabras de garantía, bigramas del título, categoría completa | 0.9457 / 0.9453 / 0.9458 vs 0.9458 | – | No |
| 29 | Re-tuneo de LightGBM con Optuna sobre v7 (40 trials) | 0.9457 vs 0.9453 (1 semilla) | – | Ver EXP30 |
| 30 | Parámetros de EXP29 con 5 semillas y en el ensamble | 0.9456 vs 0.9457; ensamble 0.9462 vs 0.9462 | – | No (ruido, como EXP17/20) |
| 31 | Reglas del conocimiento aplicadas a mano (palabras clave, tipo de publicación, stock) | la mejor gana 4 ítems de 70.000 | – | No |
| 32 | Transformer fine-tuneado: ELECTRA small en español sobre título + garantía | 0.8394 (solo) | – | Base para EXP33–34 |
| 33 | ELECTRA en el modelo: como mezcla final / como feature del LightGBM tuneado | 0.9462 = 0.9462 / **0.9465** vs 0.9458 | – | Como feature, sí |
| 34 | Ensamble v8 (= v7 + p_electra) con semillas | **0.9466** vs 0.9462 | **0.94704 (sub07)** | Poco (+0.0004, mejora en 3 de 5 folds) |
| 35 | Segundo ELECTRA "rico" (título + precio, tipo, envío, cantidades como texto) como feature extra / como mezcla | 0.8991 solo; LGBM tuneado 0.9451 vs 0.9464 / mezcla 0.9466 = 0.9466 | – | No (empeora en los 5 folds) |
| 36 | TabPFN v2 (transformer preentrenado para tablas) como modelo extra | 0.9360 solo; stacking con v8 0.9465 / promedio 0.9463 vs 0.9466 | – | No |
| 37 | TabPFN a fondo: más contexto, contexto por vecinos, TabPFN 3.5, TabPFN como combinador, features crudas (3000 filas del fold 0) | mejor: 3.5 fast con 40000 filas 0.9427; combinador 0.9467 vs v8 0.9463 | – | No |
| 38 | TabPFN 3.5 fast con features simples (3000 filas del fold 0): base / + palabras / + agregados de vendedor | TabPFN 0.914 / 0.919 / 0.924 vs LightGBM 0.900 / 0.906 / 0.919; mezcla 0.9473 / 0.9470 / 0.9460 vs 0.9463 | – | Prometedor en la muestra |
| 40 | TabPFN 3.5 fast, base + palabras, 5 folds y test (contexto 20000) | 0.9227 solo; mezcla 3:1 0.9467 / stacking 0.9467 vs 0.9466 | – | No (ruido) |
| 41 | La predicción de EXP40 como feature del LightGBM tuneado (3 semillas) | 0.9452 vs 0.9465 | – | No (empeora en 4 de 5 folds) |
| 42 | TabPFN 3.5 completo en Kaggle (GPU, todo el fold como contexto), sets rico y simple, sumados al ensamble 1:1:1:1:1 | rico 0.9463 / simple 0.9380 solos; **ensamble 0.9480** vs 0.9466 | **0.94838 (sub08)** | **Sí (+0.0014, mejora en los 5 folds)** |
| 43 | Segunda tanda de TabPFN en Kaggle: rico + 200 palabras, rico y simple con 8 estimadores, simple + agregados de vendedor | solos 0.9460 / 0.9466 / 0.9373 / 0.9400; ensamble con promedios por familia 0.9480 = sub08 | – | No (meseta) |
| 44 | TabPFN 3.5 fine-tuneado por fold (Kaggle GPU), set rico | 0.9463 solo (= sin fine-tuning, corr 0.9987); en el ensamble 0.9480–0.9481 = sub08 | – | No |
| 45 | Un LLM abierto (Qwen2.5 7B, 4 bits, 4 ejemplos) opina en las filas dudosas (p de sub08 entre 0.3 y 0.7) | en la franja: LLM 0.545 vs ensamble 0.621 (AUC 0.565); promedio 1:1 0.9442 / stacking 0.9478 vs 0.9480 | – | No (casi al azar en los dudosos) |

**Cómo se valida.** Todos los experimentos usan la misma partición: `StratifiedKFold(5, shuffle=True, random_state=42)`. La métrica es la accuracy media de los 5 folds (± desvío entre folds). En las tres submissions subidas, el score público quedó entre 0.0005 y 0.002 **por encima** de la validación y respetó el orden, así que la validación es confiable para elegir modelos sin gastar submissions.

**Formato de submission** (descubierto probando, no había archivo de ejemplo): columnas `ID,condition`, con `ID` = número de fila del test empezando en 1 y `condition` = `new`/`used`. Primero se probó `id` con el id de MercadoLibre (Kaggle pidió `ID` en mayúsculas) y después `ID` = id de MercadoLibre (los valores no coincidían).

---

## 00 · Análisis exploratorio
Notebook: `00_EDA.ipynb`

- Varias columnas son listas o diccionarios (dirección del vendedor, envío, fotos, variaciones, atributos, medios de pago, tags); hay que aplanarlas.
- Algunas columnas son constantes o casi vacías (`site_id`, `listing_source`, `international_delivery_mode`, `differential_pricing`, `subtitle`, `catalog_product_id`, `original_price`) y no aportan.
- **Lo que más separa las clases:**
  - Tipo de publicación: `free` es 93% usado, `gold_pro` es 100% nuevo, `gold_special` 98% nuevo.
  - Cantidad inicial: los usados tienen casi siempre cantidad 1 (97% contra 47% de los nuevos).
  - Vendidos: los nuevos venden mucho más (4,4 contra 0,1 en promedio).
  - Precio: los usados son más baratos en promedio.
- **Solapamiento train/test:** el 72% de las filas de test son de vendedores que aparecen en train y el 95% de categorías vistas en train. Esto anticipa que la historia del vendedor y la categoría van a servir mucho.

## 01 · Baseline: regresión logística
Notebook: `01_baseline_logreg.ipynb`

- **Idea:** un modelo rápido y simple con las variables más obvias del EDA, para tener una referencia.
- **Features (8):** log del precio, cantidad inicial = 1, cantidad vendida, garantía vacía, cantidad de fotos, relist automático (escaladas), y tipo de publicación y modo de compra (one-hot).
- **Resultado:** **0.8177 ± 0.0039.**
- **Conclusión:** con muy poco ya se acierta el 82%. Los coeficientes confirman lo del EDA: `free` y cantidad = 1 empujan a usado; `gold_special`, vender mucho y relist automático empujan a nuevo.

## 02 · LightGBM con features tabulares
Notebook: `02_lgbm_tabular.ipynb` · código de features: `src/features.py`

- **Idea:** pasar a un modelo de árboles (captura no linealidades e interacciones sin preparar nada) y aprovechar muchas más columnas.
- **Features (65):** precio (crudo, log, redondo), cantidades y ratio vendido/inicial, conteos (fotos, variaciones, atributos, tags, medios de pago), flags (MercadoPago, relist, video, tienda oficial, item padre, garantía), largo y mayúsculas del título, fechas (duración, hora y día de publicación, demora de actualización), tamaño de fotos, envío (retiro, gratis, modo, dimensiones), un flag por cada medio de pago y tag, y categóricas (tipo de publicación, estado, provincia, prefijo de categoría…). Se dejaron afuera vendedor, ciudad y categoría completa por tener demasiados valores distintos.
- **Hiperparámetros:** 600 árboles, learning rate 0.05, 63 hojas, subsample 0.8, colsample 0.8.
- **Resultado:** **0.9069 ± 0.0022** (+0.089 sobre el baseline).
- **Conclusión:** el salto grande viene del modelo de árboles y de usar más información.

## 03 · Meta-features: modelo de texto y target encoding
Notebook: `03_lgbm_texto_te.ipynb` · código: `src/meta.py`

- **Idea:** sumar dos fuentes que LightGBM no puede usar bien directamente:
  - el **texto** del título y la garantía;
  - **vendedor** y **categoría**, que tienen miles de valores distintos.
- **Qué se hizo:**
  - `p_text`: probabilidad de usado que da una regresión logística sobre TF-IDF del título (palabras 1-2 y caracteres 2-5) y de la garantía.
  - `te_seller`, `te_cat`: target encoding suavizado (proporción de usados del vendedor y de la categoría, mezclada con el promedio general), más la cantidad de ítems de cada uno.
  - Todo se calcula **out-of-fold anidado**: dentro de cada fold de validación se hacen 5 folds internos, así ninguna fila ve su propio label. Sin esto el target encoding filtra la respuesta y la validación sale inflada.
- **Hiperparámetros LightGBM:** 800 árboles, learning rate 0.03, 63 hojas, colsample 0.7.
- **Resultado:** **0.9378 ± 0.0022** (+0.031). El modelo de texto solo da 0.8397. **Público: 0.93971.**
- **Conclusión:** gran aporte. `p_text`, `te_cat` y `te_seller` quedan entre las features más importantes.

## 04 · Features nuevas (v2)
Notebook: `04_features_v2.ipynb` · código: `src/features_v2.py`

- **Idea:** explorar campos que todavía no usábamos y armar features por grupos, midiendo el aporte de cada grupo por separado (ablación).
- **Grupos (57 features):**
  - **fotos (7):** el id de cada foto termina en mes y año de subida (por ejemplo `_052015`). Con eso sale la antigüedad de las fotos respecto a la publicación (una foto vieja sugiere un ítem usado o republicado), si la foto pertenece a otra publicación, el prefijo numérico de la foto y el número de la descripción.
  - **atributos (10):** grupos de atributos (ficha técnica, vehículo, inmueble, part number) y totales de las variaciones (vendidos, disponibles, fotos, código del vendedor).
  - **título (21):** palabras clave (usado, nuevo, sellado, original, impecable, antiguo/colección, libro, disco, moneda, oferta, lote, repuesto, talle, "para reparar"…), cantidad de dígitos, signos de exclamación y palabras en mayúsculas.
  - **precio por categoría (6):** precio relativo a la mediana de su categoría, ranking dentro de la categoría, tamaño y dispersión de la categoría, cantidad media y % de publicaciones `free` en la categoría.
  - **vendedor (13):** cantidad de ítems, categorías distintas, precio medio, % con cantidad 1, % `free`/`bronze`/`gold_special`, % con MercadoPago y con garantía, tienda oficial, títulos repetidos.
  - Los agregados por vendedor y categoría se calculan con train y test juntos, **sin usar el target**.
- **Resultado (ablación, cada grupo sumado solo sobre EXP03):**

  | Grupo | Accuracy CV |
  |---|---|
  | EXP03 (base) | 0.9378 |
  | + fotos | 0.9387 |
  | + atributos | 0.9385 |
  | + título | 0.9395 |
  | + precio por categoría | 0.9384 |
  | + vendedor | 0.9398 |
  | **+ todos** | **0.9424 ± 0.0014** |

- **Público: 0.94304.**
- **Conclusión:** todos los grupos suman un poco y juntos más que por separado; la mejora aparece en los 5 folds. Los más fuertes son vendedor y título.

## 05 · Ajuste de hiperparámetros con Optuna
Notebook: `05_optuna_lgbm.ipynb` · código: `src/tune_lgbm.py` · estudio: `data/optuna_exp05.db`

- **Idea:** buscar una configuración mejor de LightGBM sobre el set de EXP04.
- **Qué se hizo:**
  - Búsqueda bayesiana (TPE) sobre 10 hiperparámetros: learning rate, cantidad de árboles, hojas, mínimo de muestras por hoja, colsample, subsample, regularización L1/L2, ganancia mínima y suavizado de categóricas.
  - Objetivo: accuracy media de la CV de 5 folds.
  - Los trials malos se cortan después de los primeros folds (MedianPruner). El primer trial es la configuración de EXP04, como referencia.
  - Se corrieron 30 trials de los 40 planeados: se cortó para liberar CPU, porque el estudio quedó guardado y la ganancia ya era chica.
- **Mejor configuración:** learning rate 0.021, 1300 árboles, 33 hojas, min_child 49, colsample 0.56, subsample 0.75, min_split_gain 0.12, cat_smooth 93. Es decir, árboles más chicos, más lentos y más regularizados.
- **Resultado:** **0.9432 ± 0.0017** contra 0.9424 de la configuración original.
- **Conclusión:** mejora chica, y algo optimista porque se mide sobre los mismos folds con los que se buscó. La configuración original ya era razonable. No se usa sola sino dentro del ensamble.

## 06 · XGBoost y CatBoost
Notebook: `06_modelos_ensamble.ipynb` · código: `src/exp06_models.py`

- **Idea:** probar otros modelos de boosting con las mismas features, para comparar y para ensamblar.
- **Qué se hizo:**
  - XGBoost (1500 árboles, learning rate 0.03, profundidad 8, categóricas nativas).
  - CatBoost (2000 iteraciones, learning rate 0.06, profundidad 8), que además usa vendedor, ciudad y categoría completa como categóricas con su propio target encoding ordenado (sin fuga).
  - Se guardan las probabilidades out-of-fold de cada modelo en `data/oof/`.
- **Resultado:** XGBoost **0.9401 ± 0.0021**, CatBoost **0.9392 ± 0.0029**; los dos por debajo de LightGBM (0.9424).
- **Conclusión:** solos no superan a LightGBM. CatBoost es el menos correlacionado con LightGBM (0.984), así que puede aportar en un ensamble.

## 07 · Ensamble
Notebook: `06_modelos_ensamble.ipynb` · código: `src/exp07_ensemble.py`

- **Idea:** combinar modelos que se equivocan en cosas distintas.
- **Qué se hizo:**
  - Promedio simple de probabilidades para las 15 combinaciones de {LGBM, LGBM tuneado, XGBoost, CatBoost}.
  - Stacking con regresión logística sobre los logits de los 4 modelos.
- **Resultado:** mejor combinación **LGBM + LGBM tuneado + CatBoost = 0.9442**. El stacking empata (0.9442); XGBoost no suma.
- **Elección:** promedio simple de esos tres, porque empata con el stacking y no tiene pesos que ajustar (menos riesgo de sobreajuste).
- **Público: 0.94495** (sub03), la mejor hasta ahora.

## Análisis de errores (sobre el ensamble de EXP07)

Se miraron los 3.908 errores out-of-fold (5,6%):

- **Vendedores con un solo ítem:** concentran el 49% de los errores, con 10,9% de error. En vendedores con más de 100 ítems el error baja a 1,4%. Ahí la historia del vendedor no ayuda y deciden el texto y los datos de la publicación.
- **Vendedores "mixtos"** (venden nuevos y usados): son el 22% de las filas pero el 39% de los errores.
- Hay más falsos "usado" (2.339) que falsos "nuevo" (1.569).
- Casos típicos:
  - "Casi Nuevo" / "como nuevo", que en realidad son usados y el modelo lee "nuevo";
  - libros y discos (ambiguos);
  - ítems de colección o antiguos publicados como nuevos.

## 08 · Predicciones de los compañeros de vendedor (descartado)
Código: `src/exp08_seller_preds.py`

- **Idea:** el vendedor es la señal más fuerte. Además del target encoding (que solo usa labels de train), usar también lo que el modelo predice para los **otros ítems del mismo vendedor, incluidos los de test**, que no tienen label.
- **Qué se hizo:**
  - Un modelo en dos etapas, todo anidado para no filtrar labels.
  - En cada fold, una 1ra etapa (LightGBM de EXP04) predice todos los ítems.
  - Para cada ítem se calcula la media de las predicciones de sus compañeros de vendedor, cuántos son y qué % predice "usado".
  - Una 2da etapa usa esas features.
  - Se probaron dos variantes de los folds internos: aleatorios, y agrupados por vendedor (para que la predicción de un compañero nunca use el label del propio ítem).
- **Resultado:** **0.9394** (aleatorios) y **0.9399** (agrupados), contra 0.9424 sin estas features.
- **Conclusión:** empeora, se descarta. La información del vendedor ya está en el target encoding. Además, estas features se ven distinto en train (predicciones de compañeros hechas sin ver al vendedor) que en validación, y la 2da etapa aprende una relación que no generaliza.

## 09 · Modelo de texto v2 y keywords del análisis de errores
Código: `src/meta.py` (`text_v2_*`), `src/features_v2.py` (`g_titulo_extra`), `src/exp09_text2.py`

- **Idea:** como la mitad de los errores son de vendedores sin historia, mejorar lo que mira la publicación en sí, sobre todo el texto.
- **Variantes del modelo de texto** (accuracy del modelo de texto solo, CV 5 folds):

  | Campos | Logística | NB-SVM |
  |---|---|---|
  | título + garantía (el de EXP03) | 0.8397 | 0.8438 |
  | + atributos y variaciones | 0.8441 | 0.8480 |
  | + categoría | 0.8544 | 0.8552 |
  | + todo (también medios de pago, tags y título 1-3 palabras) | 0.8560 | **0.8578** |

  NB-SVM es una logística sobre TF-IDF donde cada término se pondera por cuánto más aparece en una clase que en la otra (log-ratio de Naive Bayes).
- **Keywords nuevas** salidas del análisis de errores: "casi nuevo / como nuevo / poco uso" (95% usados), "sin uso / a estrenar", "en caja / blister", "importado / stock", "editorial / tapa dura", "nuevo" sin "casi", y de la garantía: "sin garantía", meses de garantía, "de fábrica/oficial".
- **Resultado en LightGBM** (`p_text2` calculado anidado por fold, sin fuga):

  | Variante | Accuracy CV |
  |---|---|
  | EXP04 (base) | 0.9424 |
  | + keywords extra | 0.9422 |
  | + `p_text2` | 0.9427 |
  | + `p_text2` reemplazando a `p_text` | 0.9422 |
  | + `p_text2` + keywords extra | 0.9428 ± 0.0014 |
  | LGBM tuneado (EXP05) + `p_text2` + keywords extra | **0.9440 ± 0.0019** (vs 0.9432) |

- **Conclusión:** el modelo de texto mejora mucho solo (0.840 → 0.858), pero dentro de LightGBM suma poco, porque gran parte de esa información (categoría, atributos, keywords) ya estaba en otras features. Las keywords nuevas solas no mueven nada (afectan a muy pocas filas). Conviene mantener `p_text` y `p_text2` juntos.

## 10 · CatBoost v3 y búsqueda de ensambles
Código: `src/exp10_cat_v3.py`

- **Idea:** pasarle también `p_text2` y las keywords nuevas a CatBoost y volver a buscar la mejor combinación de modelos.
- **Resultado:**
  - CatBoost v3: **0.9394** (antes 0.9392).
  - Probando todas las combinaciones de 2 a 4 modelos entre 7 candidatos (LGBM, LGBM tuneado, CatBoost, XGBoost y sus versiones v3), la mejor da **0.9449**. Pero hay más de 10 combinaciones entre 0.9446 y 0.9449.
- **Conclusión:** elegir la mejor entre ~100 combinaciones por diferencias de 0.0003 es sobreajustar la validación (la "ganadora" puede serlo por suerte). En vez de elegir así, conviene reducir el ruido de cada modelo (EXP11).

## 11 · Promedio de semillas y umbral
Código: `src/exp11_seeds.py`

- **Idea:** cada entrenamiento de un modelo de boosting depende de la semilla (subsample, colsample). Promediar varias semillas baja la varianza de las predicciones y da una mejora más confiable que elegir combinaciones.
- **Qué se hizo:**
  - LGBM v3 y LGBM tuneado v3 con 5 semillas cada uno; CatBoost v3 con 3 semillas (es más lento).
  - Ensamble con pesos iguales, fijados de antemano (1:1:1).
  - Se revisó el umbral de decisión entre 0.46 y 0.54.
- **Resultado:**

  | Modelo | 1 semilla | Varias semillas |
  |---|---|---|
  | LGBM v3 | 0.9428 | 0.9433 |
  | LGBM tuneado v3 | 0.9440 | 0.9441 |
  | CatBoost v3 | 0.9394 | 0.9407 |
  | **Ensamble 1:1:1** | – | **0.9447** |

  Con pesos 1:2:2 da 0.9451, pero esos pesos se eligieron mirando el resultado, así que no se usan. El umbral es plano (0.9446 a 0.9449 entre 0.46 y 0.54): se deja en 0.5.
- **Conclusión:** mejora chica pero estable: +0.0005 sobre el ensamble de la sub03, sin elegir nada mirando la validación. Es la base de la sub04, que dio **0.94576** en el público (+0.0008 sobre la sub03).

## Chequeos de "fugas legítimas" (sin resultado)

Antes de seguir con modelos se buscaron vínculos directos entre train y test que permitieran copiar el label:
- **Item padre:** ningún `parent_item_id` de test coincide con un id de train.
- **Fotos y descripción compartidas:** 1 foto y 1 descripción en común. Los ids parecen anonimizados.
- **Mismo vendedor y mismo título exacto:** 320 ítems de test (1%); en train esos grupos son 97,5% puros, pero el modelo ya los acierta por el target encoding del vendedor.
- **Orden de las filas:** el label no tiene autocorrelación con el orden (P(y_i = y_i+1) = 0.5025 contra 0.5027 al azar), así que el orden no filtra nada.

Conclusión: no hay atajos; la mejora tiene que venir del modelo.

## 12 · kNN de títulos
Código: `src/meta.py` (`knn_*`), `src/cache_knn.py`, `src/exp12_knn.py`

- **Idea:** una señal de texto distinta al modelo lineal. Para cada ítem se buscan los títulos más parecidos de train (similitud coseno sobre TF-IDF de palabras y caracteres) y se calcula la proporción de usados entre los 5, 20 y 50 vecinos (ponderada por similitud), más la similitud y el label del vecino más cercano. Anidado por fold, como el resto de las meta-features.
- **Resultado:** el kNN solo da 0.798. En LightGBM: config EXP04 0.9428 → **0.9431**; tuneado 0.9440 → **0.9439**.
- **Conclusión:** no aporta; la información de "títulos parecidos" ya la capturan el modelo de texto y el target encoding de categoría.

## 13 · Pseudo-labeling
Código: `src/exp13_pseudo.py`

- **Idea:** usar los 30.000 ítems de test sin label. Un modelo entrenado con el train del fold predice test; los ítems con predicción muy segura se suman al entrenamiento con esa etiqueta y se reentrena. Sin fuga: las features de test de cada fold se calculan solo con el train de ese fold.
- **Umbrales probados:** p < 0.02 o > 0.98 (agrega 65% del test), 0.05 (77%), 0.10 (84%) y 0.20 (91%).
- **Resultado:** sin pseudo-labeling 0.9440; con pseudo-labeling **0.9433 a 0.9437** según el umbral.
- **Conclusión:** empeora. Los ítems que se agregan son justamente los fáciles (el modelo ya los acierta), así que no enseñan nada nuevo, y los pocos errores que se cuelan con etiqueta equivocada meten ruido.

## 14 · Target encoding de interacciones del vendedor
Código: `src/exp14_te_inter.py`

- **Idea:** atacar a los vendedores "mixtos" (venden nuevos y usados), que son el 22% de las filas pero el 39% de los errores. Un vendedor puede vender nuevos en una categoría y usados en otra, o nuevos en `gold_special` y usados en `free`. Se agrega target encoding (anidado, suavizado) de: vendedor × tipo de publicación, vendedor × categoría, vendedor × prefijo de categoría, vendedor × cantidad = 1, vendedor × decil de precio, y categoría × tipo de publicación.
- **Resultado:** config EXP04 0.9428 → **0.9432**; tuneado 0.9440 → **0.9446** (mejora en 3 de 5 folds).
- **Conclusión:** mejora chica pero con sentido. Se incorpora a la versión v4 de los modelos.

## 15 · Modelos v4 con semillas
Código: `src/exp15_v4_seeds.py` · submission: `src/sub05.py`

- **Idea:** repetir el esquema de EXP11 (LGBM, LGBM tuneado y CatBoost, varias semillas, pesos iguales) con las features v4 = v3 + target encoding de interacciones del vendedor (EXP14).
- **Resultado:**

  | Modelo | v3 (EXP11) | v4 |
  |---|---|---|
  | LGBM (5 semillas) | 0.9433 | 0.9436 |
  | LGBM tuneado (5 semillas) | 0.9441 | 0.9447 |
  | CatBoost (3 semillas) | 0.9407 | 0.9413 |
  | **Ensamble 1:1:1** | 0.9447 | **0.9451** |

  Por fold, el ensamble v4 mejora en 4 de 5. Promediando v3 y v4 da 0.9453, pero se elige v4 solo por simplicidad (es un único conjunto de features para explicar y reproducir).
- **Conclusión:** mejora chica pero consistente. Es la base de la sub05, que dio **0.94671** en el público (+0.0010 sobre la sub04): mejor actual.

## 16 · Modelo lineal grande (diversidad)
Código: `src/exp16_linear_big.py`

- **Idea:** sumar al ensamble un modelo de otra familia, que se equivoque en cosas distintas que los árboles. Es una regresión logística sobre una matriz dispersa enorme: TF-IDF de título (palabras y caracteres), garantía, atributos y variaciones, más one-hot de vendedor, categoría, ciudad y todas las categóricas, más todas las numéricas discretizadas en deciles. No necesita meta-features.
- **Resultado:** **0.9290** con C = 2 (0.9270 con C = 0.5). Es muy bueno para un modelo lineal y es el menos correlacionado con los árboles (0.96). Pero sumado al ensamble v4 no mejora: con peso 0.1 a 0.3 da 0.9449 a 0.9452, y con stacking 0.9449 a 0.9451, contra 0.9451 sin él.
- **Conclusión:** descartado. Lo que sabe el modelo lineal ya lo saben los árboles a través de las meta-features (el modelo de texto y el target encoding son, en el fondo, modelos lineales y de conteo).

## 18 · CatBoost mejorado (título como texto nativo)
Código: `src/exp18_catboost.py`

- **Idea:** CatBoost nunca se había tuneado y puede procesar texto de forma nativa (`text_features`: tokeniza, arma diccionarios y estima probabilidades por palabra dentro de su propio esquema ordenado). Se le pasa el título en minúsculas además de las features v4, y se prueban profundidades distintas. Una semilla por configuración.
- **Resultado:**

  | Configuración | CV |
  |---|---|
  | v4 sin texto (referencia, depth 8, lr 0.06, 2000 it) | 0.9401 |
  | + título como texto nativo | 0.9416 |
  | + texto, depth 6, lr 0.04, 3000 it, l2 = 5 | 0.9423 |
  | + texto, depth 10, lr 0.05, 2000 it, l2 = 10 | **0.9429** |

  CatBoost mejora casi 3 milésimas y queda cerca del LGBM tuneado. Pero en el ensamble no suma: reemplazando el CatBoost v4 (3 semillas) por el de depth 10 (1 semilla) da 0.9449, y por el promedio de las tres variantes con texto 0.9452, contra 0.9451 del ensamble v4. La razón es que al mirar el título se parece más a los LGBM (correlación 0.993 con el LGBM tuneado, contra 0.987 del CatBoost original): gana precisión pero pierde diversidad.
- **Conclusión:** no cambia la submission. Queda como alternativa si se arma un ensamble con modelos más distintos.

## 19 · Precio relativo a los títulos más parecidos
Código: `src/exp19_price_nn.py` · vecinos cacheados en `data/nn_title_top30.npz` · features en `data/X_pricenn_all.pkl`

- **Idea:** un usado suele costar menos que un nuevo con el mismo título. Para cada ítem de train y test se buscan los 30 títulos más parecidos (coseno sobre TF-IDF de palabras y caracteres, sin usar etiquetas, así que no hay riesgo de fuga) y se compara su precio con el de ellos: diferencia de log-precio con el vecino más cercano, con el promedio ponderado y la mediana de los 5 y 30 más cercanos, posición del precio dentro de los vecinos, lo mismo restringido a la misma categoría, y cantidad de casi-duplicados (similitud > 0.9). Solo cuentan vecinos con la misma moneda y similitud > 0.3.
- **Resultado:** LGBM tuneado v4 (semilla 42) **0.9446 con y sin** las 19 features nuevas (por fold, sube en 2, baja en 3).
- **Conclusión:** descartado. Las features de precio por categoría y por vendedor de v2 (`price_vs_cat_median`, `sel_mean_logprice`) ya capturan esa señal; además las categorías son tan finas que pocos ítems tienen vecinos de la misma categoría.

## 17 · Re-tuneo de LightGBM con Optuna sobre v4
Código: `src/exp17_tune_v4.py` · estudio en `data/optuna_exp17.db` · notebook: `notebooks/11_catboost_texto_optuna_v4_precio.ipynb`

- **Idea:** los hiperparámetros del LGBM tuneado salieron de EXP05, con las features de ese momento (v1 + meta). Desde entonces se agregaron ~80 features (v2, texto v2, keywords, TE de interacciones), así que el óptimo podría haberse movido. Se corre Optuna (TPE, semilla 17, poda por mediana) 50 trials sobre v4, arrancando por los parámetros de EXP05 y sumando `max_bin` a la búsqueda.
- **Resultado:** 38 trials completos y 12 podados. El mejor (trial 34: lr 0.014, 1800 árboles, 36 hojas, colsample 0.47) da **0.9449** contra 0.9446 de los parámetros de EXP05 (trial 0), con una semilla.
- **Conclusión:** la diferencia (+0.0003) es del tamaño del ruido y además es el máximo de 50 intentos, así que está sesgada hacia arriba. Se confirma en EXP20.

## 20 · Confirmación de EXP17 con 5 semillas
Código: `src/exp20_tuned17_seeds.py`

- **Idea:** si la mejora de EXP17 es real, tiene que sobrevivir al promediar 5 semillas y notarse en el ensamble.
- **Resultado:**

  | | Parámetros EXP05 | Parámetros EXP17 |
  |---|---|---|
  | LGBM tuneado, 5 semillas | 0.9447 | 0.9448 |
  | Ensamble v4 1:1:1 | **0.9451** | 0.9449 |

- **Conclusión:** la mejora desaparece. Es un buen ejemplo de por qué no conviene elegir mirando el máximo de muchos intentos: con 50 trials, el mejor siempre parece mejor que la referencia aunque no lo sea. Se mantiene sub05.

## Estado al 8/10 (mediodía)
Después de sub05 se probaron cuatro ideas (EXP17 a EXP20) y ninguna mejora el ensamble: el modelo parece estar cerca de su techo con este enfoque, con mejoras recientes de 3 a 5 diezmilésimas de CV. sub05 (CV 0.9451, público 0.94671) queda como candidato a modelo final.

## 21 · Validación adversarial (diagnóstico)
Código: `src/exp21_adversarial.py`

- **Idea:** entrenar un LightGBM que intente distinguir ítems de train de ítems de test usando todas las features. Si lo logra, test tiene una distribución distinta y convendría corregirlo (pesos, validación más parecida a test). También sirve para responder si "forzar un poco de sobreajuste" podría ayudar.
- **Resultado:** AUC **0.498**, es decir, train y test son indistinguibles (la partición fue al azar). Las features "más usadas" por el clasificador son solo ruido.
- **Conclusión:** la CV de 5 folds es una muestra fiel de test, lo que explica por qué el público sigue a la CV. Por la misma razón, sobreajustar a train no puede ayudar: la complejidad que maximiza la CV (ya buscada con Optuna, que exploró hasta 200 hojas y 5000 árboles) es la que maximiza el accuracy esperado en test. Elegir modelos mirando el público tampoco conviene: el público es una parte de test, y elegir por él es sobreajustar a esa parte.

## 22 · Datos externos: embeddings de títulos con un modelo preentrenado
Código: `src/exp22_embeddings.py` · modelo en `data/minilm/` · embeddings en `data/emb_minilm.npy`

- **Qué datos externos se consideraron:** el dataset original de MercadoLibre (MLA 100k) está publicado con etiquetas e incluye a los ítems de test, así que usarlo, aunque fuera solo para entrenar, sería trampa: se descartó. Tipos de cambio (casi no hay precios en dólares) y la jerarquía de categorías de la API de MeLi (bloqueada, y el prefijo de categoría ya la aproxima) no justificaban el esfuerzo. Lo único con chance real era conocimiento del idioma: un modelo de lenguaje preentrenado.
- **Idea:** nuestro modelo de texto aprende solo de los 70.000 títulos de train. Un modelo preentrenado con millones de frases conoce sinónimos y palabras raras. Se usó `paraphrase-multilingual-MiniLM-L12-v2` (versión ONNX cuantizada, ~120 MB, descargada por Agustin porque Hugging Face está bloqueado en el entorno) para convertir cada título en un vector de 384 dimensiones (mean pooling, normalizado), corriendo en CPU con onnxruntime (~8 minutos para 100.000 títulos).
- **Resultado:**
  - Logística sobre los embeddings, sola: **0.7249** (el TF-IDF de títulos da ~0.85).
  - LGBM tuneado v4 + `p_emb` (logística anidada por fold, como `p_text`), 3 semillas: **0.9444** vs 0.9447 sin ella.
  - LGBM tuneado v4 + 32 componentes principales de los embeddings, 3 semillas: **0.9443**.
- **Conclusión:** descartado. El modelo está entrenado para detectar paráfrasis, así que sus vectores describen **qué producto es** ("celular Samsung") y no **en qué estado está**; "usado impecable" y "nuevo en caja" quedan muy cerca (similitud 0.71). La información de estado está en palabras puntuales, que el TF-IDF supervisado ya captura mejor.

## Ronda del 2/11: siete participantes nos pasaron
El líder del público tiene 0.95404 y hay siete por encima de sub05 (0.94671). La diferencia es grande (7 milésimas), así que se volvió a revisar el dataset campo por campo, buscando información que el modelo no estuviera viendo.

## 23 · Features v5: fotos, título editado, stock y fechas
Código: `src/features_v5.py`, `src/exp23_fotos_geo.py` · features en `data/X_v5_all.pkl`

- **Idea:** cuatro grupos de señales nuevas. (1) **Geometría de fotos:** las fotos sacadas con cámara o celular son 4:3 o 3:4, y las de catálogo suelen ser cuadradas. Cuando todas las fotos son 4:3, el 70% de los ítems es usado (contra 46% en general). (2) **Título editado:** el slug del permalink se genera con el título original, así que si no coincide con el título actual, la publicación se editó. (3) **Brecha de stock:** `inicial − disponible − vendidos` debería dar 0. Cuando no da 0 (15% de los ítems), el vendedor repuso stock, y ahí solo el 12% es usado, contra 53% en el resto. El modelo tenía las tres cantidades por separado pero no la resta. (4) **Fechas:** diferencia entre creación e inicio, y entre la última actualización y el fin.
- **Resultado:** solo con fotos y título editado: 0.9446 (nada). Con los cuatro grupos: **0.9450** vs 0.9447 (LGBM tuneado, 3 semillas).
- **Conclusión:** la geometría de las fotos ya la capturaba el área máxima de foto que teníamos; lo nuevo que suma es la brecha de stock.

## 24 · Target encoding de "firmas" del vendedor
Código: `src/exp24_te_mas.py`

- **Idea:** extender EXP14 (el último salto grande). Un vendedor suele usar textos, fotos y condiciones distintas para lo nuevo y lo usado: texto exacto de garantía (global y por vendedor), vendedor × tipo de fotos (fracción 4:3), × modo de envío, × medios de pago, × primera palabra del título, × brecha de stock.
- **Resultado:** **0.9452** vs 0.9450.
- **Conclusión:** suma poco pero en la dirección correcta; se mantiene.

## 25 · Palabras del título como columnas del LightGBM
Código: `src/exp25_words.py`

- **Idea:** hasta ahora el título entraba al LightGBM resumido en una sola probabilidad (`p_text`, `p_text2`). Así el modelo no puede combinar palabras con el resto de la información (por ejemplo, "iphone" con un precio bajo o con fotos de celular). Se agregan como columnas binarias las palabras más frecuentes del título (vocabulario por frecuencia, sin mirar el target).
- **Resultado:** 1000 palabras: **0.9458** (mejora en los 5 folds); 3000 palabras: 0.9457.
- **Conclusión:** sirve, y es la mejora más clara de la ronda. Se quedan 1000 palabras (igual resultado, más rápido).

## 26 · Ensamble v7 con semillas
Código: `src/exp26_v7_ens.py` · submission: `src/sub06.py`

- **Idea:** repetir el esquema de sub05 con las features v7 = v4 + EXP23 + EXP24 + EXP25. El CatBoost no recibe las 1000 palabras (sería muy lento); en su lugar usa el título como texto nativo (EXP18).
- **Resultado:**

  | Modelo | v4 (sub05) | v7 |
  |---|---|---|
  | LGBM config EXP04 (5 semillas) | 0.9436 | 0.9452 |
  | LGBM tuneado (5 semillas) | 0.9447 | 0.9457 |
  | CatBoost (3 semillas) | 0.9413 | 0.9440 |
  | **Ensamble 1:1:1** | 0.9451 | **0.9462** |

  Por fold: v7 [0.9450 0.9436 0.9484 0.9454 0.9486] vs v4 [0.9444 0.9437 0.9451 0.9449 0.9471].
- **Conclusión:** mejora de 1.1 milésimas, la más grande desde sub04 → sub05. Es la base de sub06.

## 27 · Vecino más parecido dentro del mismo vendedor
Código: `src/exp27_seller_knn.py`

- **Idea:** un vendedor mixto suele vender usados y nuevos en productos distintos (celulares usados, fundas nuevas). Para cada ítem se buscan los títulos más parecidos entre los ítems etiquetados del mismo vendedor (anidado como el TE). Donde hay vecino (69% de los ítems), la etiqueta del más parecido acierta el 92%.
- **Resultado:** LGBM tuneado v7 + estas features: 0.9457 vs 0.9458.
- **Conclusión:** descartado: el target encoding del vendedor y sus cruces (EXP14, EXP24) ya cubren esa información.

## 28 · Más texto crudo como columnas
Código: `src/exp28_mas_texto.py`

- **Idea:** seguir la línea de EXP25 con otras fuentes de texto: 300 palabras de la garantía, 300 bigramas del título, y `category_id` completa como categórica nativa del LightGBM (hasta ahora solo entraba su prefijo y su target encoding).
- **Resultado:** LGBM tuneado v7, 3 semillas: garantía 0.9457, bigramas 0.9453, categoría 0.9458, contra 0.9458 sin ellas.
- **Conclusión:** descartado. La garantía ya entra por el modelo de texto y su target encoding, y los bigramas agregan columnas ruidosas.

### Resultado público de sub06 (0.94657) y por qué no cambia la decisión
sub06 dio 0.94657 en el público, 0.00014 por debajo de sub05, aunque su CV es 0.0011 mayor. Se revisó el pipeline (mismas 1187 columnas que en la CV, mismas distribuciones en train y test) y no hay error. sub06 cambia solo 226 predicciones respecto de sub05; en la parte pública caen entre 70 y 110. En la CV, en las filas donde v4 y v7 discrepan, v7 acierta el 56%, lo que da una ganancia esperada de unas 10 filas (+0.001) con un desvío de ±0.0008. Lo observado (1 o 2 filas menos) es mala suerte dentro de lo esperable. Se sigue confiando en la CV (70.000 ítems) para decidir; para la selección final conviene marcar sub06 (mejor CV) y sub05 (mejor público).

## 29 y 30 · Re-tuneo de LightGBM sobre v7
Código: `src/exp29_tune_v7.py` (estudio `data/optuna_exp29.db`), `src/exp30_tuned29_seeds.py`

- **Idea:** con más de 1000 columnas nuevas (palabras), el óptimo de `colsample_bytree` y compañía podría haberse movido. Optuna, 40 trials, rango de colsample ampliado hacia abajo (0.05 a 0.8).
- **Resultado:** el mejor trial (colsample 0.27, 37 hojas, lr 0.031, 800 árboles) da 0.9457 contra 0.9453 de los parámetros viejos con una semilla, pero con 5 semillas queda 0.9456 contra 0.9457, y el ensamble no cambia (0.9462).
- **Conclusión:** igual que EXP17/20, la mejora era el máximo de 40 intentos ruidosos. Se mantienen los parámetros de EXP05.

## 31 · Reglas del conocimiento
- **Idea:** un participante de más arriba del ranking sugirió usar "reglas del conocimiento". El modelo ya tiene esas señales como features (palabras clave en título y garantía, tipo de publicación, stock); se probó si alguna regla aplicada a mano, por encima del modelo, acierta más que él en los ítems donde aplica (OOF del ensamble v7).
- **Resultado:**

  | Regla | Ítems | Acierto regla | Acierto modelo |
  |---|---|---|---|
  | Título dice "usado" → usado | 600 | 98.3% | 97.7% |
  | Título dice "nuevo" → nuevo | 3.246 | 80.8% | 95.2% |
  | Título dice "sellado" → nuevo | 205 | 72.2% | 96.1% |
  | Garantía de fábrica → nuevo | 3.114 | 98.9% | 99.5% |
  | gold_special / gold_pro → nuevo | 2.147 | 98.4% | 99.9% |
  | Stock repuesto → nuevo | 10.809 | 88.4% | 96.5% |
  | Cantidad inicial > 10 → nuevo | 6.398 | 97.6% | 99.0% |

- **Conclusión:** solo "título dice usado" le gana al modelo, por 4 ítems (+0.00006). Las reglas fijas fallan más que el modelo por el ruido del lenguaje ("como nuevo", libros "sellados" usados).

## 32 · Transformer fine-tuneado (ELECTRA small en español)
Código: `src/exp32_electra.py` · modelo en `data/electra/` · predicciones en `data/oof/electra_e3_oof.npy` y `electra_e3_test.npy`

- **Idea:** en EXP22 se usó un modelo preentrenado solo para sacar vectores, sin entrenarlo, y describía el producto y no su estado. Esta vez el transformer se **entrena** (fine-tuning) para distinguir nuevo de usado. Se eligió `mrm8488/electricidad-small-discriminator` (ELECTRA small en español, 14 millones de parámetros, 55 MB) porque sin GPU un modelo grande tardaría días. Agustin bajó los pesos a su compu (Hugging Face está bloqueado en el entorno) y PyTorch se instaló desde PyPI.
- **Cómo:** entrada = título [SEP] garantía (WordPiece del modelo, en minúsculas, conservando tildes, hasta 48 tokens). 3 épocas, lr 1e-4 con warmup y decaimiento lineal, batch 64. Con los mismos 5 folds: cada fold entrena con 4/5 y predice el quinto (OOF) y todo test. La predicción de test es el promedio de los 5 modelos. ~35 minutos por fold en 4 núcleos.
- **Resultado:** accuracy sola **0.8394** (folds 0.833 a 0.847), por debajo del modelo de texto TF-IDF (~0.85), pero con correlación de solo 0.79 con el ensamble: se equivoca en cosas distintas.

## 33 · Cómo sumar ELECTRA
Código: `src/exp33_electra_ens.py`

- **(a) Mezcla final:** stacking logístico sobre logit(ensamble v7) y logit(ELECTRA), ajustado en 4 folds y aplicado al quinto: **0.9462**, igual que v7 solo (le da peso 0.07 contra 0.91).
- **(b) Como feature del LightGBM tuneado v7** (3 semillas): **0.9465** vs 0.9458, mejora en 4 de 5 folds. Así el LightGBM puede combinar la opinión del transformer con el precio, el vendedor, las fotos, etc.
- **Ojo:** a diferencia de `p_text` (anidado), `p_electra` es OOF de un solo nivel, porque el anidado costaría 25 entrenamientos (~15 horas). Es el stacking estándar, pero la CV puede quedar levemente optimista.

## 34 · Ensamble v8 con semillas
Código: `src/exp34_v8_ens.py` · submission: `src/sub07.py`

- **Resultado:**

  | Modelo | v7 (sub06) | v8 (+ p_electra) |
  |---|---|---|
  | LGBM config EXP04 (5 semillas) | 0.9452 | 0.9458 |
  | LGBM tuneado (5 semillas) | 0.9457 | 0.9464 |
  | CatBoost con título nativo (3 semillas) | 0.9440 | 0.9439 |
  | **Ensamble 1:1:1** | 0.9462 | **0.9466** |

  Por fold: v8 [0.9466 0.9450 0.9469 0.9473 0.9474] vs v7 [0.9450 0.9436 0.9484 0.9454 0.9486].
- **Conclusión:** mejora chica (+0.0004) y despareja entre folds (sube en 3, baja en 2), con el posible optimismo de (b). Es candidata (sub07), pero no hay garantía de que mejore en el público.


### Resultado público de sub07: 0.94704
Mejor público hasta ahora (+0.00033 sobre sub05, +0.00047 sobre sub06). Con esto se pasa del 8.º al 5.º puesto (empatado) del leaderboard al 3/11. Confirma que el transformer fine-tuneado como feature aporta.

## 35 · Segundo transformer con más contexto ("ELECTRA rico")
Código: `src/exp32_electra.py` con `VARIANT=rico SEED=1 MAXLEN=64` · ensamble: `src/exp35_v9_ens.py` · mezcla: stacking logístico sobre logit(v8) y logit(rico)

- **Idea:** el ELECTRA de EXP32 solo veía el título (y la garantía). Le paso como segunda oración un texto armado con los datos tabulares: `precio 1500 ARS | gold_special | envío gratis | cantidad 10 vendidos 3 | buy_it_now | garantía ...`. Así el transformer puede cruzar las palabras del título con el precio y el tipo de publicación, algo que el LightGBM solo ve por separado.
- **Resultado del transformer solo:** **0.8991** OOF (folds 0.900 / 0.900 / 0.896 / 0.899 / 0.901), muy por encima del 0.839 del ELECTRA de título solo. Casi 5 horas de CPU.
- **Como feature extra del ensamble (v9 = v8 + p_electra_rico):** empeora. LGBM tuneado 5 semillas **0.9451** vs 0.9464 en v8, LGBM EXP04 **0.9447** vs 0.9458; baja en los 5 folds en ambos. Se cortó antes de terminar el CatBoost (reinicio del entorno) y no hacía falta seguir: con dos de los tres modelos peores, el ensamble no puede mejorar.
- **Como mezcla al final:** el stacking logístico le da peso 0.03 contra 0.91 y queda en **0.9466**, igual a v8. Los errores se solapan mucho: de los 3735 errores de v8, 2629 también los comete el ELECTRA rico.
- **Por qué empeora siendo mucho mejor solo (hipótesis):** el ELECTRA rico ya "sabe" lo mismo que el LightGBM (precio, tipo, cantidades), así que no trae información nueva, y como es una feature muy fuerte el LightGBM se apoya mucho en ella. Pero es OOF de un nivel: en entrenamiento los valores vienen de 4 modelos ELECTRA distintos y en validación de un quinto, con calibraciones que no coinciden. Con una feature débil eso no importa; con una fuerte, sí. El ELECTRA de título solo aporta justamente porque trae algo distinto (cómo está escrito el título) y pesa menos.
- **Conclusión:** descartado. No se genera sub08; sub07 sigue siendo el mejor candidato.

## 36 · TabPFN
Código: `src/exp36_tabpfn.py` · pesos: `data/tabpfn/tabpfn-v2-classifier.ckpt`

- **Idea (sugerida por un compañero del leaderboard):** TabPFN es un transformer preentrenado con millones de tablas sintéticas que clasifica "en contexto": recibe filas de entrenamiento etiquetadas y predice las nuevas sin entrenar pesos. Tiene un sesgo inductivo distinto a los árboles, así que podría equivocarse en otras filas y sumar al ensamble.
- **Cómo:** pesos públicos de TabPFN v2 (las versiones 2.5 en adelante piden aceptar una licencia en HuggingFace, bloqueado desde el entorno). Por límite de tamaño (pensado para hasta ~10000 filas y 500 columnas) y por correr en CPU: por fold, las 100 features v8 más importantes según un LightGBM rápido del propio fold y 10000 filas de contexto al azar. Unos 13 minutos por fold.
- **Prueba previa:** con 5000 filas de contexto daba 0.931 y con 10000, 0.9385 en 6000 filas del fold 0: más contexto ayuda, pero cada duplicación cuesta más del doble de tiempo.
- **Resultado:** **0.9360** solo (folds 0.936 / 0.933 / 0.936 / 0.939 / 0.935), lejos del 0.9464 de un LightGBM. Correlación 0.983 con el ensamble v8. Mezclado: stacking logístico **0.9465** (le da peso negativo, −0.10 contra 1.03) y promedio 1:1:1:1 **0.9463**, ambos contra 0.9466 de v8.
- **Conclusión:** descartado. Con 70000 filas los árboles le ganan con holgura a un modelo que solo puede mirar 10000 a la vez, y como usa las mismas features (incluidas las meta-features de texto y vendedor) se equivoca en las mismas filas.

## 37 · TabPFN a fondo
Código: `src/exp37_tpfn_scale.py`, `src/exp37_tpfn_knn.py`, `src/exp37_tpfn35.py`, `src/exp37_tpfn_stack.py` · pesos 3.5 fast: `data/tabpfn/tabpfn-v3.5-fast-20260909.safetensors` (Agustin aceptó la licencia en HuggingFace y los bajó a TP3\tabpfn)

- **Motivación:** el compañero del leaderboard insistió en que TabPFN "es por ahí". En EXP36 lo habíamos limitado a 10000 filas de contexto y a la versión 2, así que se exploraron sus palancas una por una. Todas las pruebas usan las mismas 3000 filas de validación del fold 0, donde el ensamble v8 acierta 0.9463.
- **Cuánto contexto (v2, 100 features):** 2500 filas → 0.932, 5000 → 0.934, 10000 → 0.938. Con 20000 se queda sin memoria (16 GB, sin GPU).
- **Contexto por vecinos cercanos (v2):** agrupar las filas a predecir en 6 clusters y darle a cada uno las 5000 filas de entrenamiento más cercanas: **0.926**, peor que 5000 al azar (0.934). En los clusters difíciles el contexto local queda demasiado homogéneo.
- **TabPFN 3.5 fast** (versión de septiembre de 2026, maneja tablas más grandes con menos memoria): 2500 → 0.930, 5000 → 0.934, 10000 → 0.938, 20000 → **0.942**, 40000 → **0.943** (25 minutos). A igual contexto rinde como la v2, pero permite usar casi todo el fold; con eso empata a un LightGBM (0.942 en esas filas) sin llegar al ensamble. Mezclado con v8 no suma: correlación 0.99, y de sus 172 errores 142 los comete también v8.
- **TabPFN como combinador del ensamble (stacking):** en vez de promediar 1:1:1, TabPFN recibe las predicciones OOF de los tres modelos de v8 y de ELECTRA (20000 filas de contexto): **0.9467** vs 0.9463, una fila de diferencia sobre 3000 (ruido). Agregándole las 20 features más importantes: 0.9460.
- **Solo features crudas** (sin las 34 meta-features de texto, vendedor, TE y ELECTRA), para que se equivoque distinto: **0.874**. Sí es distinto (correlación 0.87), pero tan débil que en la mezcla resta (0.9460).
- **Conclusión:** descartado. TabPFN funciona bien (con contexto grande empata a un LightGBM), pero sobre nuestras features aprende lo mismo que los árboles y no corrige sus errores. Si a otros les da un salto grande, probablemente parten de features más simples, donde un modelo fuerte "de caja" vale más. No se probó el 3.5 normal (24 capas en vez de 8; el archivo de 876 MB supera el límite de 400 MB para traerlo), pero con una correlación de 0.99 es muy poco probable que cambie la conclusión.

## 38 · TabPFN con features simples
Código: `src/exp38_simples.py`

- **Idea:** el primero del leaderboard (con bastante diferencia) le dijo a Agustin que usa TabPFN. Como en EXP37 TabPFN sobre nuestras features aprendía lo mismo que los árboles, la hipótesis es que rinde más sobre features simples, sin las meta-features (target encodings, predicciones de texto, ELECTRA). Mismas 3000 filas del fold 0 (v8 acierta 0.9463), contexto 20000, comparado contra un LightGBM entrenado con todo el fold sobre el mismo set.
- **Resultado:**

  | Set | Columnas | TabPFN 3.5 fast | LightGBM | Correlación con v8 | v8 + TabPFN 3:1 |
  |---|---|---|---|---|---|
  | (a) base de `features.py` | 68 | 0.9143 | 0.9000 | 0.945 | 0.9473 |
  | (b) base + 40 palabras del título (chi2) | 108 | 0.9190 | 0.9060 | 0.951 | 0.9470 |
  | (c) (b) + agregados no supervisados de vendedor y categoría | 123 | 0.9240 | 0.9193 | 0.957 | 0.9460 |

- **Conclusión:** con features simples TabPFN le gana con claridad a LightGBM (+0.013 a +0.014) y se equivoca en filas más distintas, así que en la muestra la mezcla suma 2-3 filas. Con features más elaboradas la ventaja se achica y la mezcla deja de sumar. Había que confirmarlo en toda la CV (EXP40).

## 39 · Notebook para correr TabPFN 3.5 completo en Kaggle (GPU)
Código: `src/exp39_export_kaggle.py` (exporta `data/kaggle_tabpfn/feat_*.npz`) · notebook: `kaggle/exp39_tabpfn35_kaggle.ipynb`

- El compañero lo corrió en Kaggle. Allí hay GPU y entra el modelo 3.5 normal (24 capas; acá solo el fast de 8) con todo el fold de entrenamiento como contexto (56000 filas). Se exportan dos sets: "rico" (las 100 features más importantes de v8, con las meta-features anidadas por fold, para que la CV sea comparable) y "simple" (= set (b) de EXP38). La notebook guarda OOF y test de cada set. Se archivó en 7 archivos de ~11 MB y se probó en chico en CPU. **Pendiente: que Agustin la corra en Kaggle.**

## 40 · TabPFN simple en toda la CV
Código: `src/exp40_tabpfn_simple_cv.py`

- **Resultado:** **0.9227** solo (folds 0.922 / 0.921 / 0.923 / 0.924 / 0.924; ~21 minutos por fold en CPU), correlación 0.950 con v8. Mezcla fija 3:1: **0.9467**; stacking logístico: **0.9467** (peso 0.14 contra 0.83), frente a 0.9466 de v8.
- **Conclusión:** la mejora de la muestra no se sostiene: +0.0001 es ruido. Test calculado igual (`data/oof/tabpfn35f_simpleb_c20000_test.npy`).

## 41 · TabPFN simple como feature
Código: `src/exp41_tabpfn_feature.py`

- **Idea:** ELECTRA tampoco sumaba como mezcla y sí como feature del LightGBM (EXP33), así que se probó lo mismo con la predicción de EXP40.
- **Resultado:** LightGBM tuneado 3 semillas **0.9452** vs 0.9465 sin ella; empeora en 4 de 5 folds. Pasa lo mismo que con el ELECTRA "rico" (EXP35): una feature OOF de un nivel bastante fuerte termina perjudicando.
- **Conclusión:** descartado. Con el modelo fast y 20000 filas de contexto, TabPFN no mejora nuestro ensamble ni sobre features simples. Queda la corrida de Kaggle (modelo completo y contexto completo) como última prueba.

## 42 · TabPFN 3.5 completo en Kaggle: sub08
Código: `kaggle/exp39_tabpfn35_kaggle.ipynb` (la corrió Agustin en Kaggle con GPU) · evaluación: `src/exp42_kaggle_eval.py` · submission: `src/sub08.py`

- **Cómo:** modelo 3.5 normal (24 capas), 4 estimadores, contexto = todo el fold de entrenamiento (56000 filas; para test, las 70000). Corrió en GPU a ~3 minutos por fold, unos 40 minutos en total para los dos sets.
- **Solos:** set rico **0.9463** (folds 0.9464 / 0.9441 / 0.9476 / 0.9464 / 0.9470), al nivel del ensamble v8 entero (0.9466) y del mejor LightGBM con 5 semillas (0.9464). Set simple **0.9380**. El salto respecto de la versión fast en CPU (0.943 y 0.923) viene de las 3 veces más capas y de usar todo el fold como contexto.
- **Mezclado con v8:**

  | Combinación | CV | Por fold |
  |---|---|---|
  | v8 (sub07) | 0.9466 | 0.9466 0.9450 0.9469 0.9473 0.9474 |
  | v8 + rico (1:1:1:1) | 0.9472 | 0.9469 0.9454 0.9475 0.9476 0.9484 |
  | v8 + simple (1:1:1:1) | 0.9474 | 0.9474 0.9456 0.9484 0.9476 0.9478 |
  | **v8 + rico + simple (1:1:1:1:1)** | **0.9480** | 0.9479 0.9464 0.9490 0.9481 0.9488 |
  | stacking logístico de los 5 | 0.9481 | 0.9486 0.9455 0.9489 0.9482 0.9491 |

- **Por qué suma:** el rico se equivoca casi en las mismas filas que v8 (correlación 0.993), pero no en todas (de sus 3758 errores, 546 v8 los acierta). El simple es más distinto (0.971) porque no usa las meta-features. Los dos juntos cubren errores diferentes.
- **Conclusión:** mejora de +0.0014, en los 5 folds y con pesos iguales fijados de antemano (sin elegir pesos por CV; el stacking da prácticamente lo mismo). Es la mayor mejora desde el ensamble v7. Se genera **sub08** = (3 × test de sub07 + TabPFN rico + TabPFN simple) / 5: 188 filas distintas de sub07. Las medias de las predicciones en test coinciden con las de OOF (0.460–0.462 vs 0.461–0.464), así que la corrida de test es consistente.

### Resultado público de sub08: 0.94838
Mejor público hasta ahora: +0.00134 sobre sub07 (0.94704), casi exactamente la mejora de CV (+0.0014). Confirma que TabPFN completo suma.

## 43 · Segunda tanda de TabPFN en Kaggle
Código: `src/exp43_export_kaggle2.py` (exporta `feat_extra.npz`), `kaggle/exp43_tabpfn35_kaggle_v2.ipynb` (la corrió Agustin, más lenta por los 8 estimadores), `src/exp43_kaggle_eval2.py`

- **Variantes:** rico_w (100 ricas + 200 palabras del título), rico_e8 y simple_e8 (8 estimadores internos y otra semilla), medio (simple + 15 agregados no supervisados de vendedor/categoría).
- **Solos:** rico_w **0.9460**, rico_e8 **0.9466**, simple_e8 **0.9373**, medio **0.9400** (vs rico 0.9463 y simple 0.9380 de EXP42). Las palabras no agregan nada sobre el set rico; más estimadores tampoco; los agregados le suben 0.002 al simple, pero lo acercan a v8 (correlación 0.977 vs 0.971).
- **Ensamble:** el candidato definido antes de mirar era sub08 con cada TabPFN reemplazado por el promedio de su familia, (3 × v8 + R + S) / 5 con R = rico, rico_w, rico_e8 y S = simple, simple_e8, medio: **0.9480**, igual que sub08. Otras combinaciones (los 6 TabPFN como modelos sueltos, v8 + R + S 1:1:1) dan 0.9480; stacking de los 7, 0.9482.
- **Conclusión:** meseta. TabPFN sin modificar ya dio lo que tenía en EXP42; las variantes son casi la misma predicción (en test cambiarían 32 filas respecto de sub08). No se genera sub09. Siguiente palanca: fine-tunear los pesos de TabPFN con nuestros datos (EXP44).

## 44 · TabPFN fine-tuneado
Código: `kaggle/exp44_tabpfn35_finetune_kaggle.ipynb` (la corrió Agustin) · evaluación: `src/exp44_eval_ft.py`

- **Idea:** TabPFN sin modificar llegó a su techo en EXP43; ajustar sus pesos con nuestros datos (`FinetunedTabPFNClassifier`), en cada fold solo con sus filas de entrenamiento (10 % para early stopping por AUC), y predecir con todo el fold como contexto.
- **Problemas de memoria:** en una T4 (14.5 GB) el fine-tuning no entró con 50000 ni con 16000 filas por paso; con 10000 entró para el set rico. El simple se quedó sin memoria en el primer fold y no se relanzó. Cada fold usó el tope de 15 minutos de fine-tuning (el early stopping nunca cortó); en total, unas 4 horas de GPU.
- **Resultado:** rico fine-tuneado **0.9463** (folds 0.9465 / 0.9442 / 0.9468 / 0.9463 / 0.9477), idéntico al rico sin fine-tuning. Correlación 0.9987 con él: de sus 3759 errores, 3535 son los mismos. En el ensamble: reemplazando al rico, 0.9480; sumándolo, 0.9481; frente a 0.9480 de sub08.
- **Conclusión:** con el learning rate por defecto (1e-5) y 15 minutos por fold, el fine-tuning casi no mueve los pesos. No se genera sub09 ni se relanza el simple: serían otras ~4 horas de GPU con muy poca chance de cambiar algo. sub08 sigue siendo la mejor.

## 45 · Un LLM opina en los casos dudosos
Código: `src/exp45_export_llm.py` (exporta filas y ejemplos, sin etiquetas), `kaggle/exp45_llm_dudosos_kaggle.ipynb` (la corrió Agustin, ~1 h en T4), `src/exp45_eval_llm.py`

- **Idea (sugerencia que le pasaron a Agustin):** donde el ensamble duda, que un LLM lea la publicación y dé su opinión. Para que sea reproducible y no un etiquetado manual de test, es un modelo abierto automático: Qwen2.5 7B Instruct en 4 bits, con la publicación como texto (título, precio, tipo de publicación, garantía, cantidades, envío, pagos, fotos, atributos) y 4 ejemplos resueltos de train fuera de la franja. La probabilidad de "usado" sale de los logits de las respuestas "nuevo" / "usado".
- **Filas:** p de sub08 entre 0.3 y 0.7: 3953 de train (ahí el ensamble acierta 0.621) y 1653 de test.
- **Reglas fijadas antes de mirar:** (a) promedio 1:1 con el ensamble dentro de la franja; (b) stacking logístico (logit ensamble, logit LLM) por folds.
- **Resultado:** el LLM acierta **0.545** en la franja (AUC 0.565, correlación 0.085 con el ensamble). Está sesgado a "usado" (dice usado en el 68 % de las filas; son 52 %) y muy extremo (mediana 0.92), pero ni corrigiendo el umbral pasa de 0.546. En ninguna sub-franja le gana al ensamble (0.45–0.55: 0.532 vs 0.558). (a) baja el CV total a **0.9442**; (b) le da peso casi nulo al LLM (0.04) y queda en **0.9478** vs 0.9480.
- **Conclusión:** en los casos donde el ensamble duda, el LLM no sabe más que el ensamble: la información que podría usar (título, garantía, precio) ya la usan los modelos, y lo que falta no está en el texto. No se genera sub09. sub08 sigue siendo la mejor.

