"""
extra_feats.py -- cheap grounding features for the RAGTruth BiGRU detector (used by model4.py).

Every feature compares an ANSWER sentence with its SOURCE document; none uses labels.
Feature groups (switch with --feat_groups, "none" reproduces v3):
  stem    stemmed match against the whole source / the retrieved evidence          (token cols 7,8)
  num     normalised numbers: "1,234" == "1234", "1.2 million" == "1200000",
          "five" == "5", "45%" == "0.45", rounding tolerance ("4.5" ~ "4.52")      (token col 9)
  date    month / weekday names not present in the source (also parses 2023-05-14) (token col 10)
  ngram   token is covered by a bigram / trigram (of stems) that occurs in source  (token cols 11,12)
  entity  capitalised multi-word entity ("New York") that never occurs in source   (token col 13)
  sbert   pretrained sentence-embedding cosine to the source sentences (max, top-3) (sentence level only)
Token features go into the network input and the per-token exact-match flags; sentence features go into
the GBM baseline AND into the network (through the same sentence-vector input that NLI features use).
"""
import os
import pickle
import re
from functools import lru_cache

import numpy as np

NF_TOK = 7                       # number of EXTRA token columns (original tok_feats has 7 -> total 14)
EV_COLS = [1, 8]                 # columns that depend on the retrieved evidence (zeroed by --ablate no_evidence)
EM_EXTRA = [7, 8, 9, 10, 11, 12, 13]   # extra exact-match columns (zeroed by --no_features)
ORIG_GROUPS = ["stem", "num", "date", "ngram", "entity", "sbert"]      # "all" (kept identical to the first release)
NEW_GROUPS = ["align", "neg", "claim", "detail"]                       # "new" (sentence level only)
ALL_GROUPS = ORIG_GROUPS + NEW_GROUPS
TOK_GROUP_COLS = {"stem": [7, 8], "num": [9], "date": [10], "ngram": [11, 12], "entity": [13]}
XF_NAMES = ["stem_cov", "stem_cov_idf", "bi_cov", "tri_cov", "n_num_unsup", "n_date_unsup",
            "n_ent_unsup", "frac_ent_unsup", "max_novel_run", "sbert_max", "sbert_top3",
            "best1_cov", "best2_cov", "n_unbound_num", "ans_neg", "neg_mismatch", "n_absolute_unsup", "n_opinion_unsup",
            "n_unit_mismatch", "ent_partial", "n_ent_new"]
XF_GROUP = {"stem": [0, 1, 8], "ngram": [2, 3], "num": [4], "date": [5], "entity": [6, 7], "sbert": [9, 10],
            "align": [11, 12, 13], "neg": [14, 15], "claim": [16, 17], "detail": [18, 19, 20]}


def parse_groups(s):
    """Comma list of group names. 'all' = the six original groups, 'new' = align,neg,claim,detail, 'none' = v3."""
    s = (s or "").strip().lower()
    if s in ("", "none"):
        return []
    out = []
    for x in [x.strip() for x in s.split(",") if x.strip()]:
        names = ORIG_GROUPS if x == "all" else NEW_GROUPS if x == "new" else [x]
        for n in names:
            if n not in ALL_GROUPS:
                raise SystemExit(f"unknown --feat_groups '{n}'; choose from {ALL_GROUPS}, 'all', 'new' or 'none'")
            if n not in out:
                out.append(n)
    return out


def name_suffix(groups):
    return f" +feats[{','.join(groups)}]" if groups else ""


# ----------------------------------------------------------------------------- stemming (light, deterministic)
def _is_word(t):
    return bool(t) and (t[0].isalnum() or t[0] == "_" or "\u0900" <= t[0] <= "\u097f")


@lru_cache(maxsize=200000)
def stem(w):
    if len(w) <= 3 or not (w.isascii() and w.isalpha()):
        return w
    out = w
    for suf, rep in (("sses", "ss"), ("ies", "y"), ("ied", "y")):
        if w.endswith(suf) and len(w) - len(suf) + len(rep) >= 3:
            out = w[:-len(suf)] + rep
            break
    else:
        if not w.endswith("ss"):
            for suf in ("ingly", "edly", "ing", "ed", "ly", "es", "s"):
                if w.endswith(suf) and len(w) - len(suf) >= 3:
                    out = w[:-len(suf)]
                    if suf in ("ing", "ed", "ingly", "edly") and len(out) >= 2 and out[-1] == out[-2] and out[-1] not in "lsz":
                        out = out[:-1]
                    break
    if out.endswith("e") and len(out) > 3:     # make / makes / making -> mak
        out = out[:-1]
    return out


# ----------------------------------------------------------------------------- numbers and dates
_NUM = re.compile(r"(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?:\s?(million|billion|trillion|thousand)\b|([kK])\b)?")
_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12, "k": 1e3}
_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
          "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
          "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
          "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100}   # "one" is too ambiguous
_WORD_RE = re.compile(r"\b(" + "|".join(_WORDS) + r")\b", re.I)


def num_mentions(text):
    """-> list of (start, end, value, tolerance, is_percent)."""
    out = []
    for m in _NUM.finditer(text):
        v = float(m.group(1).replace(",", "") + (m.group(2) or ""))
        sc = _SCALE.get((m.group(3) or m.group(4) or "").lower(), 1.0)
        dec = len(m.group(2)) - 1 if m.group(2) else 0
        pct = text[m.end():m.end() + 2].lstrip()[:1] == "%"
        out.append((m.start(), m.end(), v * sc, 0.5 * (10.0 ** -dec) * sc, pct))
    for m in _WORD_RE.finditer(text):
        out.append((m.start(), m.end(), float(_WORDS[m.group(1).lower()]), 0.0, False))
    return out


def _supported(v, tol, arr):
    if len(arr) == 0:
        return False
    i = int(np.searchsorted(arr, v - tol - 1e-9))
    return i < len(arr) and arr[i] <= v + tol + 1e-9


_MONTHS = {"january": 1, "february": 2, "march": 3, "april": 4, "june": 6, "july": 7, "august": 8,
           "september": 9, "october": 10, "november": 11, "december": 12, "jan": 1, "feb": 2, "mar": 3,
           "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12}   # "may" is a modal verb: skipped
_WEEK = {d: i + 1 for i, d in enumerate(["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])}
_DATE_NUM = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b|\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")


def tok_atom(raw):
    if raw[:1].isupper():
        lw = raw.lower()
        if lw in _MONTHS:
            return ("m", _MONTHS[lw])
        if lw in _WEEK:
            return ("w", _WEEK[lw])
    return None


def date_atoms(text):
    out = set()
    for w in re.findall(r"[A-Za-z]+", text):
        a = tok_atom(w)
        if a:
            out.add(a)
    for m in _DATE_NUM.finditer(text):
        if m.group(1):
            out.add(("m", int(m.group(2))))
        else:
            for g in (m.group(4), m.group(5)):
                if int(g) <= 12:
                    out.add(("m", int(g)))
    return out


# ----------------------------------------------------------------------------- lexicons for the sentence-level groups
_STOP = set("""a an the of in on at to for and or but is are was were be been being it its this that these those with by from as has have had
will would can could may might shall should they their them he she his her him we you i me my our your which who whom whose what when where
there here than then so such also very into over under about up down out off per not no nor if while""".split())
_NEG = re.compile(r"\b(not|no|never|none|neither|nor|cannot|without|nobody|nothing)\b|n't\b")
_ABSOLUTE = set("""all always every everyone everything entire entirely completely totally definitely certainly guaranteed only best worst
first last largest smallest highest lowest biggest greatest most least never ever sole solely exclusively""".split())
_OPINION = set("""excellent great amazing wonderful fantastic perfect delicious friendly popular famous renowned recommended recommend beautiful
ideal outstanding impressive terrible awful poor superb lovely stunning exceptional remarkable cozy charming vibrant""".split())
_UNIT = {"%": "pct", "percent": "pct", "km": "km", "kilometer": "km", "kilometers": "km", "kilometre": "km", "kilometres": "km",
         "mile": "mile", "miles": "mile", "mi": "mile", "m": "m", "meter": "m", "meters": "m", "metre": "m", "metres": "m",
         "ft": "ft", "feet": "ft", "foot": "ft", "cm": "cm", "inch": "in", "inches": "in", "kg": "kg", "kilogram": "kg", "kilograms": "kg",
         "lb": "lb", "lbs": "lb", "g": "g", "gram": "g", "grams": "g", "dollar": "usd", "dollars": "usd", "usd": "usd", "euro": "eur",
         "euros": "eur", "eur": "eur", "rupee": "inr", "rupees": "inr", "inr": "inr", "year": "year", "years": "year", "month": "month",
         "months": "month", "week": "week", "weeks": "week", "day": "day", "days": "day", "hour": "hour", "hours": "hour", "hr": "hour",
         "hrs": "hour", "minute": "minute", "minutes": "minute", "min": "minute", "seconds": "second", "\u00b0c": "c", "\u00b0f": "f"}
_CURR = {"$": "usd", "\u20ac": "eur", "\u00a3": "gbp", "\u20b9": "inr"}
_UNIT_AFTER = re.compile(r"\s?(%|\u00b0[cCfF]|[A-Za-z]+)")


def unit_of(text, s, e):
    """Canonical unit attached to the number mention text[s:e] (currency symbol before, or unit word after), else None."""
    if s > 0 and text[s - 1] in _CURR:
        return _CURR[text[s - 1]]
    m = _UNIT_AFTER.match(text[e:e + 14])
    return _UNIT.get(m.group(1).lower()) if m else None


def content_stems(toks):
    return frozenset(stem(t) for t in toks if _is_word(t) and t not in _STOP)


# ----------------------------------------------------------------------------- per-source context
def build_ctx(segs, tok_re):
    stems, bi, tri, nums, dates, ent, words, unitmap, seginfo = set(), set(), set(), [], set(), [], set(), {}, []
    for seg in segs:
        low = seg.lower()
        toks = tok_re.findall(low)
        st = [stem(t) for t in toks]
        stems.update(st)
        words.update(toks)
        bi.update(zip(st, st[1:]))
        tri.update(zip(st, st[1:], st[2:]))
        ments = num_mentions(seg)
        sv = []
        for s, e, v, _, pct in ments:
            nums.append(v)
            sv.append(v)
            if pct:
                nums.append(v / 100.0)
            unitmap.setdefault(round(v, 6), set()).add(unit_of(seg, s, e) or "")
        dates |= date_atoms(seg)
        ent.append(" ".join(t for t in toks if _is_word(t)))
        seginfo.append((content_stems(toks), bool(_NEG.search(low)), np.array(sorted(set(sv)), dtype=np.float64)))
    return dict(stems=stems, bi=bi, tri=tri, nums=np.array(sorted(set(nums)), dtype=np.float64), dates=dates,
                ent=" | " + " | ".join(ent) + " | ", words=words, unitmap=unitmap, segs=seginfo)


# ----------------------------------------------------------------------------- per-sentence features
def sent_feats(seg, ms, raw, low, ctx, ev, idf, maxidf):
    """seg: sentence text; ms: regex matches of the (truncated) tokens in seg; raw/low: token strings.
    Returns (X [n,7] token columns, xf [21] sentence vector; entries 9 and 10 are filled later by sbert)."""
    n = len(raw)
    st = [stem(t) for t in low]
    evst = set(stem(t) for t in ev)
    punct = np.array([not _is_word(r) for r in raw])
    cont = ~punct
    X = np.zeros((n, NF_TOK), np.float32)
    for i in range(n):
        X[i, 0] = 1.0 if punct[i] else float(st[i] in ctx["stems"])
        X[i, 1] = float(st[i] in evst)
    # numbers (and units)
    n_num = n_unit = 0
    mentions = num_mentions(seg)
    sup_ments = []
    for s, e, v, tol, pct in mentions:
        hit = [i for i, m in enumerate(ms) if m.start() < e and m.end() > s]
        if not hit:
            continue
        ok = _supported(v, tol, ctx["nums"]) or (pct and _supported(v / 100.0, tol / 100.0, ctx["nums"]))
        if not ok:
            X[hit, 2] = 1.0
            n_num += 1
        else:
            sup_ments.append((v, tol))
        u = unit_of(seg, s, e)
        if u:
            units = ctx["unitmap"].get(round(v, 6))
            if units and u not in units and (units - {""}):
                n_unit += 1
    # dates
    n_date = 0
    for i in range(n):
        a = tok_atom(raw[i])
        if a and a not in ctx["dates"]:
            X[i, 3] = 1.0
            n_date += 1
    # n-gram coverage (punctuation counts as covered, like the in-context flag)
    bi, tri = ctx["bi"], ctx["tri"]
    for i in range(n):
        if punct[i]:
            X[i, 4] = X[i, 5] = 1.0
            continue
        X[i, 4] = float((i > 0 and (st[i - 1], st[i]) in bi) or (i + 1 < n and (st[i], st[i + 1]) in bi))
        X[i, 5] = float(any(j >= 0 and j + 2 < n and (st[j], st[j + 1], st[j + 2]) in tri for j in (i - 2, i - 1, i)))
    # entities: runs of capitalised word tokens separated only by whitespace
    n_ent = n_ent_bad = n_ent_new = 0
    part = []
    i = 0
    while i < n:
        if punct[i] or not raw[i][:1].isupper():
            i += 1
            continue
        j = i
        while (j + 1 < n and not punct[j + 1] and raw[j + 1][:1].isupper() and seg[ms[j].end():ms[j + 1].start()].strip() == ""):
            j += 1
        if not (j == i and i == 0):                      # skip a lone sentence-initial capital
            n_ent += 1
            if f" {' '.join(low[i:j + 1])} " not in ctx["ent"]:
                X[i:j + 1, 6] = 1.0
                n_ent_bad += 1
                cov = float(np.mean([(t in ctx["words"]) or (stem(t) in ctx["stems"]) for t in low[i:j + 1]]))
                part.append(cov)
                n_ent_new += int(cov == 0.0)
        i = j + 1
    # sentence-level vector (original block)
    ncnt = max(int(cont.sum()), 1)
    w = np.array([min(idf.get(t, maxidf), maxidf) for t in low]) * cont
    run = best = 0
    for i in range(n):
        run = run + 1 if (cont[i] and X[i, 0] == 0) else 0
        best = max(best, run)
    # alignment with the best source sentence(s)
    A = content_stems(low)
    An = frozenset(x for x in A if not x.isdigit())
    b1 = b2 = 0.0
    n_unbound = 0
    ans_neg = float(bool(_NEG.search(seg.lower())))
    neg_mis = 0.0
    segs = ctx["segs"]
    if A and segs:
        ov = [len(A & s_[0]) for s_ in segs]
        k = int(np.argmax(ov))
        b1 = ov[k] / len(A)
        rest = A - segs[k][0]
        extra = max((len(rest & s_[0]) for s_ in segs), default=0)
        b2 = (ov[k] + extra) / len(A)
        if b1 >= 0.5:
            neg_mis = float(bool(ans_neg) != segs[k][1])
        need = min(2, len(An))
        for v, tol in sup_ments:
            if need == 0:
                break
            if not any(len(An & s_[0]) >= need and _supported(v, tol, s_[2]) for s_ in segs):
                n_unbound += 1
    # unsupported absolute / opinion words
    n_abs = sum(1 for t in low if t in _ABSOLUTE and t not in ctx["words"] and stem(t) not in ctx["stems"])
    n_op = sum(1 for t in low if t in _OPINION and t not in ctx["words"] and stem(t) not in ctx["stems"])
    xf = np.array([(X[:, 0] * cont).sum() / ncnt, (w * X[:, 0]).sum() / max(w.sum(), 1e-6),
                   (X[:, 4] * cont).sum() / ncnt, (X[:, 5] * cont).sum() / ncnt,
                   np.log1p(n_num), np.log1p(n_date), np.log1p(n_ent_bad), n_ent_bad / max(n_ent, 1),
                   np.log1p(best), 0.0, 0.0,
                   b1, b2, np.log1p(n_unbound), ans_neg, neg_mis, np.log1p(n_abs), np.log1p(n_op),
                   np.log1p(n_unit), float(np.mean(part)) if part else 1.0, np.log1p(n_ent_new)], dtype=np.float32)
    return X, xf


# ----------------------------------------------------------------------------- pretrained sentence embeddings
def load_source_segments(data_dir, flatten, split_sents):
    import json
    out = {}
    for l in open(os.path.join(data_dir, "source_info.jsonl"), encoding="utf8"):
        d = json.loads(l)
        out[d["source_id"]] = [seg.strip() for line in flatten(d["source_info"]) for _, _, seg in split_sents(line) if seg.strip()]
    return out


def attach_sbert(data, src_segments, model_name, cache_path, names=("train", "val", "test")):
    """Fills xf[9] (max cosine to any source sentence) and xf[10] (mean of the top-3). Returns False if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:                                   # noqa: BLE001
        print(f"[WARN] sentence-transformers not usable ({e}); sbert group disabled. pip install sentence-transformers")
        return False
    texts = set()
    for nm in names:
        sents, R = data[nm]
        texts.update(s["text"] for s in sents)
        for sid in set(r["sid"] for r in R):
            texts.update(src_segments.get(sid, []))
    cache = {"model": model_name, "texts": [], "emb": np.zeros((0, 1), np.float16)}
    if cache_path and os.path.exists(cache_path):
        c = pickle.load(open(cache_path, "rb"))
        if c.get("model") == model_name:
            cache = c
    index = {t: i for i, t in enumerate(cache["texts"])}
    new = sorted((t for t in texts if t not in index), key=len)
    if new:
        print(f"sbert: encoding {len(new)} new sentences with {model_name} (cached: {len(index)}) -- first run is slow on CPU")
        model = SentenceTransformer(model_name, device="cpu")
        model.max_seq_length = 128
        e = model.encode([t[:600] for t in new], batch_size=128, normalize_embeddings=True, show_progress_bar=True).astype(np.float16)
        cache["emb"] = e if len(cache["texts"]) == 0 else np.concatenate([cache["emb"], e])
        cache["texts"] = cache["texts"] + new
        index = {t: i for i, t in enumerate(cache["texts"])}
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            pickle.dump(cache, open(cache_path, "wb"))
    emb = cache["emb"]
    sid_idx = {}
    for nm in names:
        sents, R = data[nm]
        for s in sents:
            sid = R[s["rid"]]["sid"]
            if sid not in sid_idx:
                sid_idx[sid] = np.array([index[t] for t in src_segments.get(sid, []) if t in index], dtype=np.int64)
            idx = sid_idx[sid]
            if len(idx) == 0:
                continue
            sims = emb[idx].astype(np.float32) @ emb[index[s["text"]]].astype(np.float32)
            s["xf"][9] = sims.max()
            s["xf"][10] = np.sort(sims)[-3:].mean()
    return True


# ----------------------------------------------------------------------------- wiring
def finalize(data, groups, names=("train", "val", "test")):
    """Applies the group switches: zeroes disabled token columns, keeps enabled sentence features,
    and appends them to s['hand'] (GBM baseline). Returns the kept sentence-feature names."""
    zero_cols = [c for g, cols in TOK_GROUP_COLS.items() if g not in groups for c in cols]
    keep = sorted(set(j for g in groups for j in XF_GROUP.get(g, [])))
    for nm in names:
        for s in data[nm][0]:
            if zero_cols:
                s["F"][:, zero_cols] = 0.0
            s["xf"] = s["xf"][keep].astype(np.float32)
            s["hand"] = np.r_[s["hand"][:7], s["xf"]].astype(np.float32)
    return [XF_NAMES[j] for j in keep]


def merge_into_nli(data, nli_dim, names=("train", "val", "test")):
    """Standardises the sentence features with TRAIN statistics and appends them to the network's sentence vector
    (s['nli']). The original NLI vector stays in s['nli0'] (the GBM uses it). Returns the new input dimension."""
    for nm in names:
        for s in data[nm][0]:
            s["nli0"] = s["nli"]
    tr = data["train"][0]
    d = len(tr[0]["xf"]) if tr else 0
    if d == 0:
        return nli_dim
    X = np.stack([s["xf"] for s in tr])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    for nm in names:
        for s in data[nm][0]:
            z = np.clip((s["xf"] - mu) / sd, -5, 5).astype(np.float32)
            s["nli"] = np.r_[s["nli0"], z].astype(np.float32) if nli_dim else z
    return nli_dim + d

if __name__ == "__main__":
    print("extra_feats.py executed successfully")
    print("Available feature groups:", ALL_GROUPS)