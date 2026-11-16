# Examen 3 · Competencia Kaggle ap-2026-q-2 · Entrega final

Clasificación de publicaciones de MercadoLibre en nuevo / usado (métrica: accuracy).
Modelo final = **sub08**: validación cruzada 0.9480 ± 0.0009, leaderboard público **0.94838**.

## Qué hay en la carpeta

| Carpeta | Contenido |
|---|---|
| `presentacion/` | Presentación de 5 minutos (pptx y pdf): baseline, selección de modelos, modelo final, limitaciones. |
| `codigo/entrenar.py` | **(b)** Entrena el modelo final y lo guarda en `modelo/`. |
| `codigo/predecir.py` | **(c)** Aplica el modelo guardado a test y escribe la submission. |
| `codigo/predecir_kaggle.ipynb` | La misma predicción como notebook de Kaggle (necesita GPU). |
| `codigo/features.py`, `meta.py`, `electra.py` | Funciones que usan los dos scripts: features, meta-features y el modelo de texto. |
| `codigo/requirements.txt` | Versiones exactas de las librerías. |
| `pesos/electra/` | Pesos preentrenados de ELECTRA small en español (mrm8488/electricidad-small-discriminator). |
| `modelo/` | El modelo final ya entrenado (lo que escribe `entrenar.py`). |
| `submission/sub08_ensamble_v8_tabpfn.csv` | El archivo subido a Kaggle. |
| `notas/` | Bitácora de los 45 experimentos (`experimentos.md`) y la guía para explicar el trabajo. |

## El modelo final en una línea

Promedio de 5 probabilidades de "usado", con umbral 0.5:
3 modelos de árboles (LightGBM ×5 semillas, LightGBM tuneado ×5, CatBoost con el título como texto ×3), que pesan 3/5,
más TabPFN 3.5 sobre 100 columnas elegidas (1/5) y TabPFN 3.5 sobre las columnas base + 40 palabras del título (1/5).
Los árboles usan como features, entre otras, meta-features out-of-fold (texto TF-IDF, NB-SVM, target encodings del vendedor) y la probabilidad de un ELECTRA fine-tuneado sobre título y garantía.

## Cómo reproducir

1. Poner `train_data.jsonlines` y `test_data.jsonlines` de la competencia en una carpeta `datos/` al lado de `codigo/`.
2. `pip install -r codigo/requirements.txt` (Python 3.13).
3. **Entrenar** (CPU, unas 5 horas en 4 núcleos; ELECTRA es la parte larga):
   `python codigo/entrenar.py --datos datos --modelo modelo --electra pesos/electra`
4. **Predecir** (necesita GPU por TabPFN, que usa las 70.000 filas de train como contexto):
   - En Kaggle: subir `codigo/` y `modelo/` como dataset, abrir `codigo/predecir_kaggle.ipynb`, agregar los datos de la competencia y los pesos de TabPFN, elegir GPU T4 y correr todo.
   - Con GPU local: `python codigo/predecir.py --datos datos --modelo modelo --tabpfn <ruta>/tabpfn-v3.5-20260909.safetensors --salida submission.csv`

Los pesos de TabPFN 3.5 (`tabpfn-v3.5-20260909.safetensors`, de Prior Labs) no se incluyen porque su licencia pide aceptarla al descargarlos.

## Reproducibilidad

- Todas las semillas están fijas (particiones, LightGBM, CatBoost, ELECTRA, TabPFN).
- ELECTRA se entrena en CPU para que sea determinístico; el script guarda cada fold y puede retomarse si se corta, con el mismo resultado.
- Las features que no usan la etiqueta (agregados por vendedor, categoría, etc.) se calculan con train + test, igual al entrenar y al predecir.
- Verificación: las features, las meta-features y el primer modelo ELECTRA reentrenados con estos scripts dan exactamente los mismos valores que la corrida original (diferencia 0.0).
