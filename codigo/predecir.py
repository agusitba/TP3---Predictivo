"""Aplica el modelo final guardado por entrenar.py a los datos de test y escribe la submission.

Uso:  python predecir.py --datos ../datos --modelo ../modelo --tabpfn ../pesos/tabpfn-v3.5-20260909.safetensors --salida ../submission.csv
Necesita GPU para TabPFN (corre en Kaggle con una T4 en unos 10 minutos); el resto corre en CPU.
Las features sin etiqueta (agregados por vendedor y categoría) se calculan con train + test, igual que al entrenar."""
import argparse, json, os, joblib
import numpy as np, pandas as pd, torch
from catboost import CatBoostClassifier
import features as F, meta as M, electra
from entrenar import matrices, set_rico, set_simple, palabras_df, SEMILLAS_LGBM, SEMILLAS_CAT
DIR = os.path.dirname(os.path.abspath(__file__))


def arboles(a, tr, te, T):
    """Probabilidad de los 3 modelos de árboles promediados: (LightGBM + LightGBM tuneado + CatBoost) / 3."""
    n = len(tr)
    d = M.campos_texto(T['tt'])
    meta_te = joblib.load(f'{a.modelo}/meta.joblib').predict({c: v[n:] for c, v in d.items()}, {c: v[n:] for c, v in T['claves'].items()})
    meta_te['p_electra'] = electra.predecir_test(te, f'{a.modelo}/electra', f'{a.modelo}/electra')
    mt = pd.concat([pd.DataFrame(np.nan, index=range(n), columns=meta_te.columns), meta_te], ignore_index=True)
    W = palabras_df(joblib.load(f'{a.modelo}/palabras.joblib'), T['tt'])
    L, C = matrices(T, mt, W)
    B, Bc = L.iloc[n:], C.iloc[n:]
    p_l = np.mean([joblib.load(f'{a.modelo}/lgbm_s{s}.joblib').predict_proba(B)[:, 1] for s in SEMILLAS_LGBM], axis=0)
    p_t = np.mean([joblib.load(f'{a.modelo}/lgbm_tuneado_s{s}.joblib').predict_proba(B)[:, 1] for s in SEMILLAS_LGBM], axis=0)
    p_c = np.mean([CatBoostClassifier().load_model(f'{a.modelo}/catboost_s{s}.cbm').predict_proba(Bc)[:, 1] for s in SEMILLAS_CAT], axis=0)
    return (p_l + p_t + p_c) / 3, L.iloc[n:], set_simple(T, W, json.load(open(f'{a.modelo}/palabras_tabpfn_simple.json')))[n:]


def tabpfn(pesos, X_ctx, y_ctx, X_te):
    """TabPFN 3.5 con todo train como contexto: 4 estimadores, semilla 0, predicción en tandas de 2000 filas."""
    from tabpfn import TabPFNClassifier
    c = None
    for modo in ['fit_with_cache', 'fit_preprocessors']:
        try:
            c = TabPFNClassifier(model_path=pesos, device='cuda', n_estimators=4, ignore_pretraining_limits=True, fit_mode=modo, random_state=0).fit(X_ctx, y_ctx)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    out, i, tanda = [], 0, 2000
    while i < len(X_te):
        try:
            out.append(c.predict_proba(X_te[i:i + tanda])[:, 1]); i += tanda
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); tanda = max(100, tanda // 2)
    del c; torch.cuda.empty_cache()
    return np.concatenate(out)


def main(a):
    tr, te = F.cargar(a.datos)
    T = F.tablas(tr, te)
    p_arb, L_te, S_te = arboles(a, tr, te, T)
    print('árboles listos', flush=True)
    ctx = np.load(f'{a.modelo}/tabpfn_contexto.npz')
    cols = json.load(open(f'{DIR}/columnas_tabpfn_rico.json'))
    p_rico = tabpfn(a.tabpfn, ctx['rico'], ctx['y'].astype(int), set_rico(L_te, cols))
    p_simple = tabpfn(a.tabpfn, ctx['simple'], ctx['y'].astype(int), S_te)
    print('TabPFN listo', flush=True)
    p = (3 * p_arb + p_rico + p_simple) / 5
    pd.DataFrame({'ID': np.arange(1, len(te) + 1), 'condition': np.where(p > 0.5, 'used', 'new')}).to_csv(a.salida, index=False)
    print('submission escrita en', a.salida, '| usados:', round(float((p > 0.5).mean()), 4))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--datos', default=f'{DIR}/../datos'); p.add_argument('--modelo', default=f'{DIR}/../modelo')
    p.add_argument('--tabpfn', default=f'{DIR}/../pesos/tabpfn-v3.5-20260909.safetensors')
    p.add_argument('--salida', default=f'{DIR}/../submission.csv')
    main(p.parse_args())
