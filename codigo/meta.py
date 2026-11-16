"""Meta-features que usan la etiqueta: dos modelos de texto y target encoding.
Para las filas de train se calculan out-of-fold (5 folds internos), así ninguna fila ve su propia etiqueta;
para test se aplican los modelos ajustados con todo train, que se guardan en modelo/."""
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def campos_texto(df):
    return {'title': df['title'].fillna('').str.lower().values, 'warranty': df['warranty'].fillna('').str.lower().values,
            'seller_id': df['seller_id'].astype(str).values, 'category_id': df['category_id'].values,
            'attr': df['attributes'].apply(lambda l: ' '.join(f"{a.get('id', '')} {a.get('value_name', '')}" for a in l)).str.lower().values,
            'var': df['variations'].apply(lambda l: ' '.join(f"{c.get('name', '')} {c.get('value_name', '')}" for v in l
                                                             for c in (v.get('attribute_combinations') or []))).str.lower().values,
            'pay': df['non_mercado_pago_payment_methods'].apply(lambda l: ' '.join(m.get('id', '') for m in l)).values,
            'tags': df['tags'].apply(' '.join).values}


class TextoTFIDF:
    """p_text: logística sobre TF-IDF del título (palabras 1-2 y caracteres 2-5) y de la garantía."""
    def fit(self, d, y):
        self.v = [TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents='unicode'),
                  TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=200000),
                  TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents='unicode')]
        A = hstack([self.v[0].fit_transform(d['title']), self.v[1].fit_transform(d['title']), self.v[2].fit_transform(d['warranty'])]).tocsr()
        self.lr = LogisticRegression(C=4, max_iter=2000).fit(A, y); return self

    def predict(self, d):
        B = hstack([self.v[0].transform(d['title']), self.v[1].transform(d['title']), self.v[2].transform(d['warranty'])]).tocsr()
        return self.lr.predict_proba(B)[:, 1]


class TextoNBSVM:
    """p_text2: NB-SVM (logística sobre TF-IDF escalado por el log-ratio de Naive Bayes) con título, garantía,
    atributos, variaciones, categoría, medios de pago y tags."""
    def fit(self, d, y):
        self.vs = [(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents='unicode'), 'title'),
                   (TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=200000), 'title'),
                   (TfidfVectorizer(ngram_range=(1, 3), min_df=2, sublinear_tf=True, strip_accents='unicode'), 'title'),
                   (TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents='unicode'), 'warranty'),
                   (TfidfVectorizer(min_df=2, sublinear_tf=True), 'attr'), (TfidfVectorizer(min_df=2, sublinear_tf=True), 'var'),
                   (CountVectorizer(token_pattern=r'\S+', binary=True, min_df=2), 'category_id'),
                   (CountVectorizer(binary=True), 'pay'), (CountVectorizer(binary=True), 'tags')]
        A = hstack([v.fit_transform(d[c]) for v, c in self.vs]).tocsr()
        p, q = 1 + A[y == 1].sum(0), 1 + A[y == 0].sum(0)
        self.r = np.asarray(np.log((p / p.sum()) / (q / q.sum()))).ravel()
        self.lr = LogisticRegression(C=4, max_iter=3000).fit(A.multiply(self.r).tocsr(), y); return self

    def predict(self, d):
        B = hstack([v.transform(d[c]) for v, c in self.vs]).tocsr()
        return self.lr.predict_proba(B.multiply(self.r).tocsr())[:, 1]


class TargetEncoding:
    """Para cada clave: proporción de usados suavizada hacia el promedio general (alpha = 10) y cantidad de apariciones."""
    def fit(self, claves, y, alpha=10):
        self.prior, self.tablas = y.mean(), {}
        for n, k in claves.items():
            s = pd.DataFrame({'k': k, 'y': y}).groupby('k')['y'].agg(['sum', 'count'])
            self.tablas[n] = ((s['sum'] + alpha * self.prior) / (s['count'] + alpha), s['count'])
        return self

    def predict(self, claves):
        out = {}
        for n, (enc, cnt) in self.tablas.items():
            k = pd.Series(claves[n])
            out[n], out[n.replace('te_', 'cnt_')] = k.map(enc).fillna(self.prior).values, k.map(cnt).fillna(0).values
        return pd.DataFrame(out)


def _sub(d, idx):
    return {c: v[idx] for c, v in d.items()}


class Meta:
    """Las cuatro piezas juntas. Columnas en el orden en que las ve el modelo."""
    def fit(self, d, claves, y):
        self.texto, self.nbsvm = TextoTFIDF().fit(d, y), TextoNBSVM().fit(d, y)
        self.te_base = TargetEncoding().fit({'te_seller': d['seller_id'], 'te_cat': d['category_id']}, y)
        self.te_claves = TargetEncoding().fit(claves, y); return self

    def predict(self, d, claves):
        b = self.te_base.predict({'te_seller': d['seller_id'], 'te_cat': d['category_id']})
        out = pd.DataFrame({'p_text': self.texto.predict(d)})
        out = pd.concat([out, b], axis=1)
        out['p_text2'] = self.nbsvm.predict(d)
        return pd.concat([out, self.te_claves.predict(claves)], axis=1)


def meta_oof(d, claves, y):
    """Meta-features de train calculadas out-of-fold con 5 folds internos."""
    parts = []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y):
        f = Meta().fit(_sub(d, a), _sub(claves, a), y[a]).predict(_sub(d, b), _sub(claves, b)); f.index = b; parts.append(f)
    return pd.concat(parts).sort_index().reset_index(drop=True)
