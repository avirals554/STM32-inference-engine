#!/usr/bin/env python3
"""
tiny_lm_suitability.py
----------------------
Profile a plain-text corpus and judge how "easy" it is for a VERY small
language model to learn. Pure standard library: no installs needed.

Usage:
    python3 tiny_lm_suitability.py path/to/text.txt
    python3 tiny_lm_suitability.py text.txt --heldout 0.15 --json

What it measures (and why it matters for a tiny model):
  * char-level held-out bits-per-char  -> raw predictability; the single
        best proxy for "can a small model compress this?"
  * word-level held-out perplexity      -> how guessable the next word is
  * gzip redundancy                     -> cheap Kolmogorov-complexity proxy;
        redundant text is simpler text
  * vocabulary size + hapax rate        -> tiny models can't memorise a long
        tail of words seen only once
  * TTR / MATTR (lexical diversity)      -> lower = more repetitive = simpler
  * avg word & sentence length          -> structural load

All thresholds are rough heuristics. The raw numbers are always shown so you
can judge for yourself; the letter grade is just a convenience.
"""

import argparse
import collections
import gzip
import json
import math
import re
import sys


# ----------------------------------------------------------------------------
# tokenisation
# ----------------------------------------------------------------------------
def words_of(text):
    # lowercase word tokens, keep apostrophes inside words (don't, l'eau)
    return re.findall(
        r"[^\W\d_]+(?:['\u2019][^\W\d_]+)*", text.lower(), flags=re.UNICODE
    )


def sentences_of(text):
    parts = re.split(r"[.!?]+(?:\s|$)", text)
    return [p for p in (s.strip() for s in parts) if p]


# ----------------------------------------------------------------------------
# information-theoretic measures
# ----------------------------------------------------------------------------
def unigram_entropy(symbols):
    """Plain Shannon entropy of the symbol distribution (bits/symbol)."""
    n = len(symbols)
    if n == 0:
        return 0.0
    counts = collections.Counter(symbols)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def char_trigram_bpc(text, heldout, k=0.1):
    """
    Fit a char trigram model on a train split, report bits-per-char on the
    held-out split. Add-k smoothing; unseen chars map to <unk>; unseen
    contexts back off to uniform automatically via the smoothing term.
    """
    chars = list(text)
    if len(chars) < 50:
        return None
    split = int(len(chars) * (1 - heldout))
    train, test = chars[:split], chars[split:]

    vocab = set(train) | {"<unk>"}
    V = len(vocab)
    train = [c if c in vocab else "<unk>" for c in train]
    test = [c if c in vocab else "<unk>" for c in test]

    ctx_counts = collections.Counter()  # P(c | c-2, c-1)
    tri_counts = collections.Counter()
    pad = ["<unk>", "<unk>"]
    seq = pad + train
    for i in range(2, len(seq)):
        ctx = (seq[i - 2], seq[i - 1])
        ctx_counts[ctx] += 1
        tri_counts[(ctx, seq[i])] += 1

    total_bits, n = 0.0, 0
    seqt = pad + test
    for i in range(2, len(seqt)):
        ctx = (seqt[i - 2], seqt[i - 1])
        c = seqt[i]
        num = tri_counts.get((ctx, c), 0) + k
        den = ctx_counts.get(ctx, 0) + k * V
        total_bits += -math.log2(num / den)
        n += 1
    return total_bits / n if n else None


def word_bigram_ppl(tokens, heldout, k=0.1):
    """Held-out word-level perplexity from an add-k bigram model."""
    if len(tokens) < 50:
        return None, None
    split = int(len(tokens) * (1 - heldout))
    train, test = tokens[:split], tokens[split:]

    vocab = set(train) | {"<unk>", "<s>"}
    V = len(vocab)
    train = ["<s>"] + [w if w in vocab else "<unk>" for w in train]
    test = ["<s>"] + [w if w in vocab else "<unk>" for w in test]

    uni = collections.Counter(train)
    bi = collections.Counter(zip(train, train[1:]))

    total_bits, n = 0.0, 0
    for a, b in zip(test, test[1:]):
        num = bi.get((a, b), 0) + k
        den = uni.get(a, 0) + k * V
        total_bits += -math.log2(num / den)
        n += 1
    if not n:
        return None, None
    bits_per_word = total_bits / n
    return 2**bits_per_word, bits_per_word


def gzip_redundancy(text):
    raw = text.encode("utf-8")
    if not raw:
        return None
    comp = gzip.compress(raw, compresslevel=9)
    return len(comp) / len(raw)  # lower = more redundant = simpler


# ----------------------------------------------------------------------------
# lexical richness
# ----------------------------------------------------------------------------
def mattr(tokens, window=100):
    """Moving-average type-token ratio: length-robust diversity."""
    if len(tokens) < window:
        return len(set(tokens)) / len(tokens) if tokens else 0.0
    ratios = []
    for i in range(len(tokens) - window + 1):
        win = tokens[i : i + window]
        ratios.append(len(set(win)) / window)
    return sum(ratios) / len(ratios)


# ----------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------
def band(value, good, ok, lower_is_better=True):
    """Return (sub_score 0-100, label) for a metric against two thresholds."""
    if value is None:
        return None, "n/a"
    if lower_is_better:
        if value <= good:
            return 100, "great"
        if value <= ok:
            # linear interp between thresholds
            return round(100 - 50 * (value - good) / (ok - good)), "ok"
        return max(0, round(50 - 50 * (value - ok) / ok)), "hard"
    else:
        if value >= good:
            return 100, "great"
        if value >= ok:
            return round(100 - 50 * (good - value) / (good - ok)), "ok"
        return max(0, round(50 * value / ok)), "hard"


def analyse(text, heldout):
    chars = text
    tokens = words_of(text)
    sents = sentences_of(text)

    n_chars = len(chars)
    n_tokens = len(tokens)
    vocab = set(tokens)
    n_vocab = len(vocab)
    counts = collections.Counter(tokens)
    hapax = sum(1 for w, c in counts.items() if c == 1)

    m = {}
    m["chars"] = n_chars
    m["words"] = n_tokens
    m["sentences"] = len(sents)
    m["vocab_size"] = n_vocab
    m["ttr"] = n_vocab / n_tokens if n_tokens else 0.0
    m["mattr_100"] = mattr(tokens, 100)
    m["hapax_rate"] = hapax / n_vocab if n_vocab else 0.0
    m["avg_word_len"] = sum(len(w) for w in tokens) / n_tokens if n_tokens else 0.0
    m["avg_sentence_len_words"] = n_tokens / len(sents) if sents else 0.0
    m["char_unigram_entropy"] = unigram_entropy(list(chars))
    m["char_bpc_heldout"] = char_trigram_bpc(text, heldout)
    ppl, bpw = word_bigram_ppl(tokens, heldout)
    m["word_ppl_heldout"] = ppl
    m["word_bits_heldout"] = bpw
    m["gzip_ratio"] = gzip_redundancy(text)

    # sub-scores: thresholds tuned for "very very small" model suitability
    subs = {}
    subs["char predictability (BPC)"] = band(m["char_bpc_heldout"], 2.5, 3.5)
    subs["redundancy (gzip ratio)"] = band(m["gzip_ratio"], 0.35, 0.50)
    subs["rare-word load (hapax rate)"] = band(m["hapax_rate"], 0.30, 0.50)
    subs["lexical repetition (MATTR)"] = band(m["mattr_100"], 0.55, 0.75)
    subs["word predictability (ppl)"] = band(m["word_ppl_heldout"], 80, 250)

    # weighted composite — BPC and redundancy matter most for a tiny model
    weights = {
        "char predictability (BPC)": 0.30,
        "redundancy (gzip ratio)": 0.25,
        "rare-word load (hapax rate)": 0.20,
        "lexical repetition (MATTR)": 0.15,
        "word predictability (ppl)": 0.10,
    }
    num = den = 0.0
    for name, (sc, _) in subs.items():
        if sc is not None:
            num += sc * weights[name]
            den += weights[name]
    composite = round(num / den) if den else None

    return m, subs, composite


def grade(score):
    if score is None:
        return "n/a"
    return next(
        g
        for t, g in [(85, "A"), (70, "B"), (55, "C"), (40, "D"), (-1, "F")]
        if score >= t
    )


# ----------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------
def fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:,.3f}" if abs(v) < 1000 else f"{v:,.1f}"
    return f"{v:,}"


def report(path, m, subs, composite):
    out = []
    out.append("=" * 64)
    out.append(f" TINY-LM SUITABILITY REPORT  —  {path}")
    out.append("=" * 64)

    if m["words"] < 2000:
        out.append("")
        out.append("  !! WARNING: under ~2,000 words. Metrics will be noisy and")
        out.append("     the held-out scores especially are not reliable.")

    out.append("")
    out.append(" CORPUS SIZE")
    out.append(f"   characters .............. {fmt(m['chars'])}")
    out.append(f"   words ................... {fmt(m['words'])}")
    out.append(f"   sentences ............... {fmt(m['sentences'])}")
    out.append(f"   vocabulary (unique) ..... {fmt(m['vocab_size'])}")

    out.append("")
    out.append(" PREDICTABILITY  (lower = simpler for the model)")
    out.append(
        f"   char unigram entropy .... {fmt(m['char_unigram_entropy'])} bits/char"
    )
    out.append(f"   char trigram BPC (held).. {fmt(m['char_bpc_heldout'])} bits/char")
    out.append(f"   word bigram perplexity .. {fmt(m['word_ppl_heldout'])}")
    out.append(f"   gzip ratio .............. {fmt(m['gzip_ratio'])}  (compressed/raw)")

    out.append("")
    out.append(" LEXICAL & STRUCTURAL")
    out.append(f"   type-token ratio ........ {fmt(m['ttr'])}")
    out.append(f"   MATTR (window=100) ...... {fmt(m['mattr_100'])}")
    out.append(
        f"   hapax rate .............. {fmt(m['hapax_rate'])}  (vocab seen once)"
    )
    out.append(f"   avg word length ......... {fmt(m['avg_word_len'])} chars")
    out.append(f"   avg sentence length ..... {fmt(m['avg_sentence_len_words'])} words")

    out.append("")
    out.append(" SUB-SCORES  (0-100, higher = easier for a tiny model)")
    for name, (sc, label) in subs.items():
        bar = "#" * (sc // 5) if sc is not None else ""
        out.append(f"   {name:<30} {fmt(sc):>5}  {label:<6} {bar}")

    out.append("")
    out.append("-" * 64)
    out.append(
        f" OVERALL SUITABILITY: {fmt(composite)} / 100   GRADE: {grade(composite)}"
    )
    out.append("-" * 64)
    out.append(verdict(composite))
    out.append("")
    return "\n".join(out)


def verdict(score):
    if score is None:
        return " Not enough text to judge."
    if score >= 85:
        return (
            " Excellent. Highly repetitive and predictable — a very small\n"
            " model should learn this with little data. Good candidate."
        )
    if score >= 70:
        return (
            " Good. Reasonably learnable for a small model. Watch the\n"
            " weakest sub-score above; trimming rare words may help."
        )
    if score >= 55:
        return (
            " Mixed. Workable, but expect a tiny model to struggle on the\n"
            " long tail. Consider more data, or simplify the vocabulary."
        )
    if score >= 40:
        return (
            " Hard. High diversity / low redundancy. A very small model\n"
            " will underfit. Increase data, restrict vocab, or use a bigger model."
        )
    return (
        " Very hard for a tiny model. Too unpredictable / too many rare\n"
        " words relative to its capacity."
    )


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Profile a .txt for tiny-LM suitability.")
    ap.add_argument("file", help="path to a UTF-8 .txt file")
    ap.add_argument(
        "--heldout",
        type=float,
        default=0.10,
        help="fraction held out for predictability tests (default 0.10)",
    )
    ap.add_argument("--json", action="store_true", help="emit raw metrics as JSON")
    args = ap.parse_args()

    try:
        with open(args.file, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError as e:
        sys.exit(f"Could not read {args.file}: {e}")

    if not text.strip():
        sys.exit("File is empty.")

    m, subs, composite = analyse(text, args.heldout)

    if args.json:
        print(
            json.dumps(
                {
                    "metrics": m,
                    "sub_scores": {k: v[0] for k, v in subs.items()},
                    "overall": composite,
                    "grade": grade(composite),
                },
                indent=2,
            )
        )
    else:
        print(report(args.file, m, subs, composite))


if __name__ == "__main__":
    main()
