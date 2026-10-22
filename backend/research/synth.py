"""Synthetic hard negatives for RAGTruth-style training data (pure python, no TF).

Takes a FAITHFUL response (no annotated hallucination span) and corrupts ONE sentence so that it is no longer supported
by the source. The corrupted span is labelled as a hallucination (char offsets), exactly like a real RAGTruth label.

Three heuristics (all deterministic given the seed):
  number   a number in the sentence is shifted to a value that does not occur anywhere in the source
  entity   a capitalised word (not sentence-initial) is replaced by another capitalised word taken from the SAME source
  negation a copula/auxiliary ("is", "was", "are", "has", ...) gets "not" inserted after it

These are heuristics, not ground truth: a corrupted sentence is occasionally still true. Use them for TRAINING only and
always evaluate on the real, human-annotated validation/test data.
"""
import hashlib
import random
import re

NUM_RE = re.compile(r"(?<![\w.])(\d{1,4}(?:,\d{3})*(?:\.\d+)?)(?![\w])")
CAP_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")
AUX_RE = re.compile(r"\b(is|was|are|were|has|have|had|can|will|does|did)\b")
SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _sentences(text):
    out, start = [], 0
    for m in SENT_RE.finditer(text):
        seg = text[start:m.start()]
        if seg.strip():
            out.append((start, m.start()))
        start = m.end()
    if text[start:].strip():
        out.append((start, len(text)))
    return out


def _num_swap(sent, s0, src_text, rng):
    ms = list(NUM_RE.finditer(sent))
    rng.shuffle(ms)
    for m in ms:
        raw = m.group(1)
        clean = raw.replace(",", "")
        try:
            val = float(clean)
        except ValueError:
            continue
        for _ in range(8):
            if "." in clean:
                new = round(val + rng.choice([-1, 1]) * rng.uniform(0.1, 3.0) * max(1.0, abs(val) * 0.1), len(clean.split(".")[1]))
                new_s = f"{new:.{len(clean.split('.')[1])}f}"
            elif 1900 <= val <= 2100:
                new_s = str(int(val) + rng.choice([-1, 1]) * rng.randint(1, 6))
            else:
                step = max(1, int(abs(val) * rng.uniform(0.1, 0.6)))
                new_s = str(max(0, int(val) + rng.choice([-1, 1]) * step))
            if new_s != clean and not re.search(rf"(?<![\w.]){re.escape(new_s)}(?![\w])", src_text):
                return s0 + m.start(1), s0 + m.end(1), new_s, "number"
    return None


def _entity_swap(sent, s0, src_text, rng):
    ms = [m for m in CAP_RE.finditer(sent) if m.start() > 0]
    rng.shuffle(ms)
    pool = sorted(set(CAP_RE.findall(src_text)))
    for m in ms:
        cands = [w for w in pool if w != m.group(1) and w not in sent]
        if cands:
            return s0 + m.start(1), s0 + m.end(1), rng.choice(cands), "entity"
    return None


def _negate(sent, s0, src_text, rng):
    ms = list(AUX_RE.finditer(sent))
    if not ms:
        return None
    m = rng.choice(ms)
    if sent[m.end():m.end() + 4].lower() == " not":
        return None
    return s0 + m.end(), s0 + m.end(), " not", "negation"


def corrupt(response, src_text, rng):
    """Returns (new_response, [label dict]) or None when no sentence can be corrupted."""
    spans = _sentences(response)
    order = list(range(len(spans)))
    rng.shuffle(order)
    ops = [_num_swap, _entity_swap, _negate]
    for si in order:
        a, b = spans[si]
        sent = response[a:b]
        if len(sent.split()) < 4:
            continue
        rng.shuffle(ops)
        for op in ops:
            r = op(sent, a, src_text, rng)
            if r is None:
                continue
            c0, c1, new, kind = r
            out = response[:c0] + new + response[c1:]
            return out, [dict(start=c0, end=c0 + len(new), label_type="Synthetic", text=new, kind=kind)]
    return None


def make_synthetic(responses, src_text_of, ratio, seed):
    """responses: iterable of response dicts (faithful ones only are used). src_text_of(sid) -> source text.
    Selection and corruption depend only on (seed, response id), so the verifier script can regenerate the same set."""
    out = []
    for r in responses:
        if r["labels"]:
            continue
        h = int(hashlib.md5(f"{seed}|{r['id']}".encode()).hexdigest()[:8], 16)
        if (h % 10000) / 10000.0 >= ratio:
            continue
        rng = random.Random(h)
        res = corrupt(r["response"], src_text_of(r["source_id"]), rng)
        if res is None:
            continue
        text, labels = res
        s = dict(r)
        s.update(id=f"{r['id']}~syn", response=text, labels=labels, synthetic=True)
        out.append(s)
    return out
