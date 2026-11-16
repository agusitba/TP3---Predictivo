"""Entrena el modelo final (sub08) y lo guarda en modelo/.

Modelo final = promedio de 5 probabilidades de "usado":
  3 modelos de árboles (LightGBM x 5 semillas, LightGBM tuneado x 5, CatBoost con título x 3), que cuentan 3/5,
  + TabPFN 3.5 sobre el set "rico" (1/5) + TabPFN 3.5 sobre el set "simple" (1/5).
TabPFN no se entrena: el "modelo" es su contexto (las filas de train y sus etiquetas), que se guarda acá y se usa en predecir.py.

Uso:  python entrenar.py --datos ../datos --modelo ../modelo --electra ../pesos/electra
  --datos: carpeta con train_data.jsonlines y test_data.jsonlines
  --electra: pesos preentrenados de mrm8488/electricidad-small-discriminator (config.json, pytorch_model.bin, vocab.txt)
Tarda unas 5 horas en una CPU de 4 núcleos (ELECTRA ~3 h, el resto ~2 h)."""
import argparse, json, os, shutil, time, joblib
import numpy as np, pandas as pd, lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import StratifiedKFold
import features as F, meta as M, electra

# hiperparámetros elegidos en la etapa de selección (EXP04 y Optuna EXP05)
LGBM_BASE = dict(n_estimators=800, learning_rate=0.03, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1)
LGBM_TUNEADO = dict(learning_rate=0.020849894291815222, n_estimators=1300, num_leaves=33, min_child_samples=49,
                    colsample_bytree=0.5609321356356409, subsample=0.7525128711041711, reg_lambda=0.0016391623737230614,
                    reg_alpha=0.015357537742737364, min_split_gain=0.11931314125454957, cat_smooth=92.8368407124553, subsample_freq=1, verbose=-1)
CATBOOST = dict(iterations=2000, learning_rate=0.06, depth=8, verbose=0, thread_count=4)
SEMILLAS_LGBM, SEMILLAS_CAT = [42, 1, 2, 3, 4], [42, 1, 2]
DIR = os.path.dirname(os.path.abspath(__file__))


def matrices(T, mt, W):
    """Matrices de los árboles para todas las filas (train + test): LightGBM con categóricas nativas y 1000 palabras;
    CatBoost con categóricas como texto, sin palabras (lee el título como texto)."""
    Xl = T['X'].drop(columns=['seller_id', 'city', 'category_id']).copy()
    for c in F.CAT_COLS:
        if c in Xl: Xl[c] = Xl[c].astype('category')
    Xc = T['X'].copy()
    for c in F.CAT_COLS: Xc[c] = Xc[c].astype(str)
    resto = [mt, T['V2'], T['EX'], T['V5']]
    L = pd.concat([Xl] + resto + [W], axis=1)
    C = pd.concat([Xc] + resto, axis=1)
    C['title_txt'] = T['tt']['title'].fillna('').str.lower().values
    return L, C


def set_rico(L, cols):
    """Set "rico" de TabPFN: las 100 columnas más importantes del ensamble de árboles (elegidas en EXP39)."""
    return L[cols].apply(pd.to_numeric, errors='coerce').astype(np.float32).values


def set_simple(T, W, palabras):
    """Set "simple" de TabPFN: las 68 columnas base (categóricas como códigos) + 40 palabras del título."""
    X = T['X'].copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]): X[c] = X[c].astype('category').cat.codes
    return pd.concat([X, W[palabras]], axis=1).astype(np.float32).values


def main(a):
    t0 = time.time(); os.makedirs(a.modelo, exist_ok=True)
    tr, te = F.cargar(a.datos); n = len(tr)
    y = (tr['condition'] == 'used').astype(int).values
    T = F.tablas(tr, te)
    print(f'features listas ({time.time() - t0:.0f}s)', flush=True)

    # meta-features: out-of-fold en train; ajustadas con todo train para test
    d = M.campos_texto(T['tt']); dtr = {c: v[:n] for c, v in d.items()}
    ctr = {c: v[:n] for c, v in T['claves'].items()}
    meta_tr = M.meta_oof(dtr, ctr, y)
    joblib.dump(M.Meta().fit(dtr, ctr, y), f'{a.modelo}/meta.joblib')
    print(f'meta-features listas ({time.time() - t0:.0f}s)', flush=True)

    # ELECTRA: 5 modelos con la partición de siempre; p_electra out-of-fold en train
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y))
    meta_tr['p_electra'] = electra.entrenar_folds(tr, y, folds, a.electra, f'{a.modelo}/electra')
    for f in ['config.json', 'vocab.txt']: shutil.copy(f'{a.electra}/{f}', f'{a.modelo}/electra/{f}')
    print(f'ELECTRA listo ({time.time() - t0:.0f}s)', flush=True)

    # 1000 palabras del título como columnas 0/1 (vocabulario de train)
    cv = CountVectorizer(min_df=30, binary=True, strip_accents='unicode', token_pattern=r'(?u)\b\w\w+\b', max_features=1000)
    cv.fit(tr['title'].fillna('').str.lower())
    joblib.dump(cv, f'{a.modelo}/palabras.joblib')
    W = palabras_df(cv, T['tt'])

    # árboles (solo filas de train; test se arma igual en predecir.py)
    mt = pd.concat([meta_tr, pd.DataFrame(np.nan, index=range(len(te)), columns=meta_tr.columns)], ignore_index=True)
    L, C = matrices(T, mt, W)
    A, Ac = L.iloc[:n], C.iloc[:n]
    for nombre, cfg in [('lgbm', LGBM_BASE), ('lgbm_tuneado', LGBM_TUNEADO)]:
        for s in SEMILLAS_LGBM:
            joblib.dump(lgb.LGBMClassifier(**cfg, random_state=s).fit(A, y), f'{a.modelo}/{nombre}_s{s}.joblib')
        print(f'{nombre} listo ({time.time() - t0:.0f}s)', flush=True)
    for s in SEMILLAS_CAT:
        CatBoostClassifier(**CATBOOST, random_seed=s).fit(Ac, y, cat_features=F.CAT_COLS, text_features=['title_txt']).save_model(f'{a.modelo}/catboost_s{s}.cbm')
    print(f'catboost listo ({time.time() - t0:.0f}s)', flush=True)

    # TabPFN: el contexto (train completo) de cada set
    cols = json.load(open(f'{DIR}/columnas_tabpfn_rico.json'))
    sc, _ = chi2(W.iloc[:n], y); palabras = list(W.columns[np.argsort(-np.nan_to_num(sc))[:40]])
    json.dump(palabras, open(f'{a.modelo}/palabras_tabpfn_simple.json', 'w'))
    np.savez_compressed(f'{a.modelo}/tabpfn_contexto.npz', y=y.astype(np.int8),
                        rico=set_rico(L.iloc[:n], cols), simple=set_simple(T, W, palabras)[:n])
    print(f'modelo guardado en {a.modelo} ({time.time() - t0:.0f}s)')


def palabras_df(cv, tt):
    return pd.DataFrame(cv.transform(tt['title'].fillna('').str.lower()).toarray().astype(np.uint8), columns=[f'w_{t}' for t in cv.get_feature_names_out()])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--datos', default=f'{DIR}/../datos'); p.add_argument('--modelo', default=f'{DIR}/../modelo')
    p.add_argument('--electra', default=f'{DIR}/../pesos/electra')
    main(p.parse_args())
