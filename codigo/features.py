"""Features sin etiqueta (no usan el target), calculadas con las mismas funciones en entrenamiento y en predicción.
- base (68 columnas, EXP02): precio, cantidades, tipo de publicación, envío, fotos, fechas, vendedor, categoría...
- v2 (57, EXP04): fotos, atributos, palabras clave del título, precio relativo a la categoría, perfil del vendedor.
- extra (9, EXP09): palabras clave del análisis de errores ("casi nuevo", "sin uso", garantía).
- v5 (20, EXP23): geometría de las fotos, título editado, brecha de stock, fechas.
Las de v2 y v5 que agregan por vendedor o categoría se calculan sobre train + test juntos (sin usar el target)."""
import re, unicodedata
import numpy as np
import pandas as pd


def cargar(dir_datos):
    tr = pd.read_json(f"{dir_datos}/train_data.jsonlines", lines=True)
    te = pd.read_json(f"{dir_datos}/test_data.jsonlines", lines=True)
    return tr, te


# ---------------------------------------------------------------- base
def _len(x):
    return len(x) if isinstance(x, (list, dict)) else 0


def build_features(df):
    """Features tabulares (sin texto). Devuelve un DataFrame numérico/categórico."""
    f = pd.DataFrame(index=df.index)
    # numéricas
    f["price"] = df["price"]
    f["log_price"] = np.log1p(df["price"])
    f["base_eq_price"] = (df["base_price"] == df["price"]).astype(int)
    f["price_round"] = (df["price"] % 1 == 0).astype(int)
    f["price_round100"] = (df["price"] % 100 == 0).astype(int)
    f["initial_quantity"] = df["initial_quantity"]
    f["sold_quantity"] = df["sold_quantity"]
    f["available_quantity"] = df["available_quantity"]
    f["qty_eq1"] = (df["initial_quantity"] == 1).astype(int)
    f["sold_ratio"] = df["sold_quantity"] / df["initial_quantity"].clip(lower=1)
    f["n_pictures"] = df["pictures"].apply(_len)
    f["n_variations"] = df["variations"].apply(_len)
    f["n_attributes"] = df["attributes"].apply(_len)
    f["n_tags"] = df["tags"].apply(_len)
    f["n_nmp_methods"] = df["non_mercado_pago_payment_methods"].apply(_len)
    f["n_deal_ids"] = df["deal_ids"].apply(_len)
    f["accepts_mercadopago"] = df["accepts_mercadopago"].astype(int)
    f["automatic_relist"] = df["automatic_relist"].astype(int)
    f["has_video"] = df["video_id"].notna().astype(int)
    f["has_official_store"] = df["official_store_id"].notna().astype(int)
    f["has_parent"] = df["parent_item_id"].notna().astype(int)
    f["has_original_price"] = df["original_price"].notna().astype(int)
    f["warranty_null"] = df["warranty"].isna().astype(int)
    f["warranty_len"] = df["warranty"].fillna("").str.len()
    f["title_len"] = df["title"].str.len()
    f["title_nwords"] = df["title"].str.split().str.len()
    f["title_upper_ratio"] = df["title"].apply(lambda s: sum(c.isupper() for c in s) / max(len(s), 1))

    # fechas
    st = pd.to_datetime(df["start_time"])
    sp = pd.to_datetime(df["stop_time"])
    dc = pd.to_datetime(df["date_created"], utc=True).dt.tz_localize(None)
    lu = pd.to_datetime(df["last_updated"], utc=True).dt.tz_localize(None)
    f["duration_days"] = (sp - st).dt.total_seconds() / 86400
    f["update_lag_h"] = (lu - dc).dt.total_seconds() / 3600
    f["start_hour"] = st.dt.hour
    f["start_dow"] = st.dt.dayofweek
    f["start_ts"] = st.astype("int64") // 10**9

    # fotos: tamaño máximo
    def pic_stats(p):
        if not p:
            return (0, 0)
        areas = []
        for x in p:
            try:
                w, h = map(int, x.get("max_size", "0x0").split("x"))
                areas.append(w * h)
            except Exception:
                pass
        return (max(areas) if areas else 0, np.mean(areas) if areas else 0)
    ps = df["pictures"].apply(pic_stats)
    f["pic_max_area"] = ps.str[0]
    f["pic_mean_area"] = ps.str[1]

    # shipping
    sh = df["shipping"]
    f["ship_local_pick_up"] = sh.apply(lambda s: int(bool(s.get("local_pick_up"))))
    f["ship_free"] = sh.apply(lambda s: int(bool(s.get("free_shipping"))))
    f["ship_mode"] = sh.apply(lambda s: s.get("mode"))
    f["ship_n_methods"] = sh.apply(lambda s: _len(s.get("methods")))
    f["ship_n_tags"] = sh.apply(lambda s: _len(s.get("tags")))
    f["ship_has_dims"] = sh.apply(lambda s: int(s.get("dimensions") is not None))

    # medios de pago no-MP (flags)
    for pid in ["MLATB", "MLAWC", "MLAMO", "MLAOT", "MLAVE", "MLAMC", "MLAAM", "MLAVS", "MLADC", "MLAMS", "MLABC", "MLAMP", "MLAWT"]:
        f[f"pay_{pid}"] = df["non_mercado_pago_payment_methods"].apply(lambda l: int(any(m.get("id") == pid for m in l)))

    # tags (flags)
    for t in ["dragged_bids_and_visits", "good_quality_thumbnail", "dragged_visits", "free_relist", "poor_quality_thumbnail"]:
        f[f"tag_{t}"] = df["tags"].apply(lambda l: int(t in l))
    f["sub_status"] = df["sub_status"].apply(lambda l: l[0] if l else "none")

    # categóricas
    f["listing_type_id"] = df["listing_type_id"]
    f["buying_mode"] = df["buying_mode"]
    f["status"] = df["status"]
    f["currency_id"] = df["currency_id"]
    f["state"] = df["seller_address"].apply(lambda a: a.get("state", {}).get("id"))
    f["city"] = df["seller_address"].apply(lambda a: a.get("city", {}).get("name"))
    f["category_id"] = df["category_id"]
    f["seller_id"] = df["seller_id"].astype(str)
    f["cat_prefix"] = df["category_id"].str[:6]
    return f


CAT_COLS = ["ship_mode", "sub_status", "listing_type_id", "buying_mode", "status", "currency_id",
            "state", "city", "category_id", "seller_id", "cat_prefix"]


# ---------------------------------------------------------------- v2 y extra
def _pic_date(pid):
    """Las fotos tienen id tipo '324401-MLA3037874345_052015': el sufijo es mes+año de subida (MMYYYY)."""
    m = re.search(r'_(\d{2})(\d{4})$', pid)
    if not m:
        return np.nan
    mm, yy = int(m.group(1)), int(m.group(2))
    return yy * 12 + mm - 1 if 1 <= mm <= 12 and 2000 <= yy <= 2016 else np.nan


def g_fotos(df):
    f = pd.DataFrame(index=df.index)
    st = pd.to_datetime(df['start_time'])
    start_m = st.dt.year * 12 + st.dt.month - 1
    dates = df['pictures'].apply(lambda l: [_pic_date(p.get('id', '')) for p in l])
    f['pic_age_max'] = start_m - dates.apply(lambda l: np.nanmin(l) if l and not all(np.isnan(l)) else np.nan)
    f['pic_age_min'] = start_m - dates.apply(lambda l: np.nanmax(l) if l and not all(np.isnan(l)) else np.nan)
    f['pic_frac_dated'] = dates.apply(lambda l: np.mean([not np.isnan(x) for x in l]) if l else np.nan)
    f['pic_prefix'] = df['pictures'].apply(lambda l: np.log1p(float(l[0]['id'].split('-')[0])) if l and l[0]['id'].split('-')[0].isdigit() else np.nan)
    f['pic_frac_own_id'] = df.apply(lambda r: np.mean([r['id'] in p.get('id', '') for p in r['pictures']]) if r['pictures'] else np.nan, axis=1)
    f['thumb_dated'] = df['thumbnail'].str.contains(r'_\d{6}-I', regex=True).astype(int)
    f['desc_num'] = df['descriptions'].apply(lambda l: float(re.search(r'-(\d+)', l[0]).group(1)) if l and re.search(r'-(\d+)', l[0]) else np.nan)
    return f


def g_atributos(df):
    f = pd.DataFrame(index=df.index)
    ids = df['attributes'].apply(lambda l: {a.get('id', '') for a in l})
    groups = df['attributes'].apply(lambda l: {a.get('attribute_group_id', '') for a in l})
    for g in ['DFLT', 'FIND']:
        f[f'attr_grp_{g}'] = groups.apply(lambda s: int(g in s))
    f['attr_vehiculo'] = ids.apply(lambda s: int(any('KMTS' in x for x in s)))
    f['attr_inmueble'] = ids.apply(lambda s: int(any('INMUEBLE' in x for x in s)))
    f['attr_part_number'] = ids.apply(lambda s: int(any('PART_NUMBER' in x for x in s)))
    v = df['variations']
    f['var_sold'] = v.apply(lambda l: sum(x.get('sold_quantity', 0) or 0 for x in l))
    f['var_avail'] = v.apply(lambda l: sum(x.get('available_quantity', 0) or 0 for x in l))
    f['var_n_pics'] = v.apply(lambda l: sum(len(x.get('picture_ids') or []) for x in l))
    f['var_custom_field'] = v.apply(lambda l: int(any(x.get('seller_custom_field') for x in l)))
    f['var_n_attr_comb'] = v.apply(lambda l: sum(len(x.get('attribute_combinations') or []) for x in l))
    return f


KEYWORDS = {
    'kw_usado': r'usad', 'kw_nuevo': r'nuev', 'kw_sellado': r'sellad|cerrad', 'kw_original': r'original',
    'kw_impecable': r'impecable|excelente|muy buen|buen estado', 'kw_garantia': r'garant', 'kw_antiguo': r'antig|vintage|retro|colecci',
    'kw_libro': r'libro|revista|novela|edici|tomo', 'kw_disco': r'\bcd\b|vinilo|\blp\b|dvd|cassette',
    'kw_moneda': r'moneda|billete|estampilla|sello postal', 'kw_oferta': r'oferta|promo|liquido|liquidacion|outlet',
    'kw_envio': r'envio|env[ií]o gratis|gratis', 'kw_lote': r'\blote\b|\bpack\b|x ?\d+ ?u', 'kw_mayor': r'por mayor|mayorista',
    'kw_repuesto': r'repuesto|\bpara\b.*\bmodelo\b|compatible', 'kw_talle': r'talle|\bt ?\d{2}\b',
    'kw_funciona': r'funcion', 'kw_detalle': r'detalle|roto|para reparar|no funciona',
}


def g_titulo(df):
    t = df['title'].fillna('').str.lower()
    f = pd.DataFrame({k: t.str.contains(p, regex=True).astype(int) for k, p in KEYWORDS.items()}, index=df.index)
    f['title_n_digits'] = t.str.count(r'\d')
    f['title_n_excl'] = df['title'].str.count('!')
    f['title_all_caps_words'] = df['title'].str.count(r'\b[A-ZÁÉÍÓÚÑ]{3,}\b')
    return f


def g_precio_cat(df):
    f = pd.DataFrame(index=df.index)
    lp = np.log1p(df['price'])
    for key in ['category_id']:
        g = lp.groupby(df[key])
        f['price_vs_cat_median'] = lp - g.transform('median')
        f['price_rank_in_cat'] = g.rank(pct=True)
        f['cat_n_items'] = g.transform('size')
        f['cat_std_price'] = g.transform('std')
    f['cat_mean_qty'] = df['initial_quantity'].groupby(df['category_id']).transform('mean')
    f['cat_frac_free'] = (df['listing_type_id'] == 'free').groupby(df['category_id']).transform('mean')
    return f


def g_vendedor(df):
    f = pd.DataFrame(index=df.index)
    s = df['seller_id']
    f['sel_n_items'] = s.map(s.value_counts())
    f['sel_n_cats'] = df.groupby('seller_id')['category_id'].transform('nunique')
    f['sel_mean_logprice'] = np.log1p(df['price']).groupby(s).transform('mean')
    f['sel_mean_qty'] = df['initial_quantity'].groupby(s).transform('mean')
    f['sel_frac_qty1'] = (df['initial_quantity'] == 1).groupby(s).transform('mean')
    f['sel_mean_sold'] = df['sold_quantity'].groupby(s).transform('mean')
    for lt in ['free', 'bronze', 'gold_special']:
        f[f'sel_frac_{lt}'] = (df['listing_type_id'] == lt).groupby(s).transform('mean')
    f['sel_frac_mp'] = df['accepts_mercadopago'].astype(int).groupby(s).transform('mean')
    f['sel_frac_warranty'] = df['warranty'].notna().groupby(s).transform('mean')
    f['sel_has_official'] = df['official_store_id'].notna().groupby(s).transform('max').astype(int)
    f['sel_n_titles_dup'] = df.groupby(['seller_id', 'title'])['id'].transform('size')
    return f


GROUPS = {'fotos': g_fotos, 'atributos': g_atributos, 'titulo': g_titulo, 'precio_cat': g_precio_cat, 'vendedor': g_vendedor}


def build_features_v2(df_all, groups=GROUPS):
    return pd.concat([fn(df_all) for name, fn in groups.items()], axis=1)


# ---------- extra (EXP09), salidas del análisis de errores ----------
KEYWORDS_EXTRA = {
    'kw_casi_nuevo': r'casi nuev|como nuev|semi ?nuev|poco uso|muy poco uso|nuevo sin uso|impecable estado',
    'kw_sin_uso': r'sin uso|sin estrenar|a estrenar',
    'kw_en_caja': r'en caja|caja cerrada|blister|en su caja',
    'kw_importado': r'importad|nacional|fabrica|stock|por mayor',
    'kw_autor': r'\b(editorial|ed\.|tapa dura|ilustrado)\b',
}


def g_titulo_extra(df):
    t = df['title'].fillna('').str.lower()
    f = pd.DataFrame({k: t.str.contains(p, regex=True).astype(int) for k, p in KEYWORDS_EXTRA.items()}, index=df.index)
    f['kw_nuevo_puro'] = (t.str.contains('nuev') & ~f['kw_casi_nuevo'].astype(bool)).astype(int)
    w = df['warranty'].fillna('').str.lower()
    f['warr_sin_garantia'] = w.str.contains(r'sin garant|no tiene garant|no posee|^no$', regex=True).astype(int)
    f['warr_meses'] = w.str.extract(r'(\d+)\s*mes')[0].astype(float)
    f['warr_fabrica'] = w.str.contains(r'fabrica|oficial|fabricante', regex=True).astype(int)
    return f


# ---------------------------------------------------------------- v5
def _wh(s):
    try:
        w, h = map(int, s.split('x'))
        return (w, h) if h > 0 and w > 0 else None
    except Exception:
        return None


def _geo(l):
    S = [x for x in (_wh(p.get('size', '')) for p in l) if x]
    M = [x for x in (_wh(p.get('max_size', '')) for p in l) if x]
    if not S:
        return [np.nan] * 12
    r = np.array([w / h for w, h in S])
    is43 = np.isclose(r, 4 / 3, atol=.02) | np.isclose(r, 3 / 4, atol=.02)
    ma = np.array([w * h for w, h in M]) if M else np.array([np.nan])
    mx = np.array([max(m) for m in M]) if M else np.array([np.nan])
    return [is43.mean(), float(is43[0]), np.isclose(r, 1, atol=.01).mean(), (r < 1).mean(),
            np.mean(mx > 500) if M else np.nan, np.nanmedian(ma), np.nanmin(ma), np.nanmax(mx),
            len(set(S)) / len(S), len(set(np.round(r, 2))), np.isclose(r, 16 / 9, atol=.03).mean() + np.isclose(r, 9 / 16, atol=.03).mean(),
            np.mean([m[0] * 3 == m[1] * 4 or m[0] * 4 == m[1] * 3 for m in M]) if M else np.nan]


GEO_COLS = ['pg_f43', 'pg_first43', 'pg_fsq', 'pg_fport', 'pg_fbig', 'pg_med_area', 'pg_min_area', 'pg_maxdim',
            'pg_uniq_size', 'pg_n_ratios', 'pg_f169', 'pg_f43_exact_max']


def _alnum(s):
    s = unicodedata.normalize('NFKD', str(s).lower()).encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z0-9]', '', s)


def build_features_v5(df):
    f = pd.DataFrame(df['pictures'].apply(_geo).tolist(), columns=GEO_COLS, index=df.index)
    slug = df['permalink'].str.extract(r'MLA\d+-(.*)-_JM')[0].fillna('')
    a, b = slug.apply(_alnum), df['title'].apply(_alnum)
    f['title_edited'] = (a != b).astype(int)
    f['slug_len_ratio'] = a.str.len() / b.str.len().clip(lower=1)
    dates = df['pictures'].apply(lambda l: [m.group(1) for m in (re.search(r'_(\d{6})$', p.get('id', '')) for p in l) if m])
    f['pic_n_dates'] = dates.apply(lambda l: len(set(l)))
    f['pic_frac_firstdate'] = dates.apply(lambda l: l.count(l[0]) / len(l) if l else np.nan)
    gap = df['initial_quantity'] - df['available_quantity'] - df['sold_quantity']
    f['qty_gap'] = gap
    f['qty_gap_nz'] = (gap != 0).astype(int)
    f['qty_avail_ratio'] = df['available_quantity'] / df['initial_quantity'].clip(lower=1)
    f['qty_restock'] = (df['available_quantity'] > df['initial_quantity']).astype(int)
    st = pd.to_datetime(df['start_time'], unit='ms') if np.issubdtype(df['start_time'].dtype, np.number) else pd.to_datetime(df['start_time'])
    sp = pd.to_datetime(df['stop_time'], unit='ms') if np.issubdtype(df['stop_time'].dtype, np.number) else pd.to_datetime(df['stop_time'])
    dc = pd.to_datetime(df['date_created'], utc=True).dt.tz_localize(None)
    lu = pd.to_datetime(df['last_updated'], utc=True).dt.tz_localize(None)
    f['start_minus_created_s'] = (st - dc).dt.total_seconds()
    f['stop_minus_updated_d'] = (sp - lu).dt.total_seconds() / 86400
    return f


# ---------------------------------------------------------------- todo junto
def tablas(tr, te):
    """Todas las features sin etiqueta, para train + test concatenados (train primero)."""
    tt = pd.concat([tr, te], ignore_index=True)
    X = pd.concat([build_features(tr), build_features(te)], ignore_index=True)
    V5 = build_features_v5(tt)
    return {'tt': tt, 'X': X, 'V2': build_features_v2(tt), 'EX': g_titulo_extra(tt), 'V5': V5, 'claves': claves_te(tt, X, V5)}


def claves_te(tt, X, V5):
    """Claves para target encoding: vendedor x tipo de publicación, categoría, precio... (EXP14) y "firmas" del vendedor:
    cómo escribe la garantía, qué fotos sube, cómo envía y cobra, primera palabra del título (EXP24)."""
    s = X['seller_id'].astype(str)
    pb = pd.qcut(X['log_price'], 10, labels=False, duplicates='drop').astype(str)
    k = {'te_sel_lt': s + '|' + X['listing_type_id'].astype(str), 'te_sel_cat': s + '|' + X['category_id'].astype(str),
         'te_sel_catp': s + '|' + X['cat_prefix'].astype(str), 'te_sel_q1': s + '|' + X['qty_eq1'].astype(str),
         'te_sel_price': s + '|' + pb, 'te_cat_lt': X['category_id'].astype(str) + '|' + X['listing_type_id'].astype(str)}
    s = tt['seller_id'].astype(str)
    w = tt['warranty'].fillna('<nulo>').str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    pics = V5['pg_f43'].fillna(-1).round(1).astype(str)
    ship = tt['shipping'].apply(lambda d: f"{d.get('mode')}|{int(bool(d.get('free_shipping')))}|{int(bool(d.get('local_pick_up')))}")
    pay = tt['non_mercado_pago_payment_methods'].apply(lambda l: ','.join(sorted(m.get('id', '') for m in l)))
    fw = tt['title'].fillna('').str.lower().str.extract(r'^(\w+)')[0].fillna('')
    k.update({'te_warr': w, 'te_sel_warr': s + '|' + w, 'te_sel_pics': s + '|' + pics, 'te_sel_ship': s + '|' + ship,
              'te_sel_pay': s + '|' + pay, 'te_sel_fw': s + '|' + fw, 'te_sel_gap': s + '|' + V5['qty_gap_nz'].astype(str), 'te_fw': fw})
    return {n: v.values for n, v in k.items()}
