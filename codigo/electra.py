"""ELECTRA small en español (mrm8488/electricidad-small-discriminator) fine-tuneado para leer título [SEP] garantía.
Su probabilidad de "usado" entra como feature (p_electra) de los modelos de árboles:
- en train, out-of-fold: 5 modelos, cada uno entrenado con 4/5 de train y aplicado al quinto restante;
- en test, el promedio de esos 5 modelos.
Todo en CPU (4 hilos) para que el resultado sea exactamente reproducible."""
import os, time
import numpy as np, torch
from tokenizers import BertWordPieceTokenizer
from transformers import ElectraConfig, ElectraForSequenceClassification, get_linear_schedule_with_warmup

EPOCHS, BS, LR, MAXLEN = 3, 64, 1e-4, 48
torch.set_num_threads(4)


def textos(df):
    return list(zip(df['title'].fillna('').tolist(), df['warranty'].fillna('sin dato').tolist()))


def codificar(pares, dir_pesos):
    tok = BertWordPieceTokenizer(f'{dir_pesos}/vocab.txt', lowercase=True, strip_accents=False)
    tok.enable_truncation(MAXLEN); tok.enable_padding(length=MAXLEN, pad_id=tok.token_to_id('[PAD]'), pad_token='[PAD]')
    E = tok.encode_batch(pares)
    return {'input_ids': torch.tensor([e.ids for e in E]), 'attention_mask': torch.tensor([e.attention_mask for e in E]),
            'token_type_ids': torch.tensor([e.type_ids for e in E])}


def predecir(m, E, idx):
    m.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(idx), 512):
            b = idx[s:s + 512]
            logits = m(input_ids=E['input_ids'][b], attention_mask=E['attention_mask'][b], token_type_ids=E['token_type_ids'][b]).logits
            out.append(torch.softmax(logits, -1)[:, 1].numpy())
    return np.concatenate(out)


def entrenar(E, y, idx, dir_pesos):
    m = ElectraForSequenceClassification.from_pretrained(dir_pesos, num_labels=2)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    steps = int(np.ceil(len(idx) / BS) * EPOCHS); sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    rng = np.random.RandomState(0); yt = torch.tensor(y); step = 0; t0 = time.time()
    while step < steps:
        perm = idx[rng.permutation(len(idx))]
        for s in range(0, len(perm), BS):
            if step >= steps: break
            b = perm[s:s + BS]; m.train()
            loss = m(input_ids=E['input_ids'][b], attention_mask=E['attention_mask'][b], token_type_ids=E['token_type_ids'][b], labels=yt[b]).loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(); step += 1
            if step % 500 == 0: print(f'    paso {step}/{steps}  {time.time() - t0:.0f}s', flush=True)
    return m


def entrenar_folds(tr, y, folds, dir_pesos, dir_salida):
    """Entrena los 5 modelos (uno por fold) y devuelve p_electra out-of-fold para train.
    Guarda cada modelo y el estado del generador aleatorio, así una corrida cortada sigue igual que una entera."""
    os.makedirs(dir_salida, exist_ok=True)
    E = codificar(textos(tr), dir_pesos)
    oof = np.zeros(len(y))
    torch.manual_seed(0)
    for k, (itr, iva) in enumerate(folds):
        f_mod, f_rng = f'{dir_salida}/fold{k}.pt', f'{dir_salida}/fold{k}_rng.pt'
        if os.path.exists(f_mod) and os.path.exists(f_rng):
            m = _cargar(dir_pesos, f_mod); torch.set_rng_state(torch.load(f_rng))
        else:
            print(f'  ELECTRA fold {k}', flush=True)
            m = entrenar(E, y, itr, dir_pesos)
            torch.save(m.state_dict(), f_mod); torch.save(torch.get_rng_state(), f_rng)
        oof[iva] = predecir(m, E, iva)
        print(f'  ELECTRA fold {k}: accuracy {((oof[iva] > .5) == y[iva]).mean():.4f}', flush=True)
    return oof


def _cargar(dir_config, f_mod):
    m = ElectraForSequenceClassification(ElectraConfig.from_pretrained(dir_config, num_labels=2))
    m.load_state_dict(torch.load(f_mod)); return m


def predecir_test(te, dir_pesos, dir_modelos):
    E = codificar(textos(te), dir_pesos)
    idx = np.arange(len(te))
    return np.mean([predecir(_cargar(dir_pesos, f'{dir_modelos}/fold{k}.pt'), E, idx) for k in range(5)], axis=0)
