"""THE shared text normalizer. Version it; import it everywhere.

Used for training labels, evaluation references, and evaluation hypotheses alike.
A mismatch between the normalizer used on labels and the one used on references is the
most common way to fool yourself in an ASR project, so there is exactly one of these.

Order is deliberate -- see rnd-docs/06-evaluation-protocol.md section 1.
"""

from __future__ import annotations

import re
import unicodedata

NORMALIZER_VERSION = "1.0.0"

# Zero-width and formatting characters that are invisible but break string equality.
_INVISIBLE = dict.fromkeys(
    [0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x00AD, 0xFEFF], None
)

_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# Non-speech annotation tags from rnd-docs/04 rule 7.
_TAG_RE = re.compile(r"\[(?:unk(?::\d+)?|overlap|laugh|music|applause|noise|foreign:[a-z]{2})\]")

# Keep intra-word hyphens and apostrophes (product-market, don't); drop everything else.
_PUNCT_RE = re.compile(r"[^\w\sঀ-৿'\-]", flags=re.UNICODE)
_EDGE_PUNCT_RE = re.compile(r"(?<!\w)['\-]+|['\-]+(?!\w)")
_WS_RE = re.compile(r"\s+")


def strip_tags(text: str) -> str:
    return _TAG_RE.sub(" ", text)


def normalize_text(text: str, *, keep_case: bool = False) -> str:
    if not text:
        return ""

    # 1. Canonical Unicode composition. Bengali conjuncts have several valid encodings.
    text = unicodedata.normalize("NFC", text)

    # 2. Remove invisible characters.
    text = text.translate(_INVISIBLE)

    # 3. Bengali digits to ASCII, so 2025 == ২০২৫.
    text = text.translate(_BN_DIGITS)

    # 4. Lowercase Latin. Bengali is caseless, so this is a no-op there.
    if not keep_case:
        text = text.lower()

    # 5. Danda to period, then strip punctuation.
    text = text.replace("।", ".").replace("॥", ".")
    text = strip_tags(text)
    text = _PUNCT_RE.sub(" ", text)
    text = _EDGE_PUNCT_RE.sub(" ", text)

    # 6. Collapse whitespace.
    return _WS_RE.sub(" ", text).strip()


# --------------------------------------------------------------- code-mixing measures

_BENGALI_RE = re.compile(r"[ঀ-৿]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def token_script(token: str) -> str:
    """'bn', 'en', or 'neutral' (digits, symbols). Mixed tokens follow their first letter."""
    has_bn = bool(_BENGALI_RE.search(token))
    has_en = bool(_LATIN_RE.search(token))
    if has_bn and has_en:
        # e.g. "deployটা" -- a Latin stem with a Bangla suffix counts as English.
        for ch in token:
            if _LATIN_RE.match(ch):
                return "en"
            if _BENGALI_RE.match(ch):
                return "bn"
    if has_bn:
        return "bn"
    if has_en:
        return "en"
    return "neutral"


def codemix_stats(text: str) -> dict:
    """CMI, switch-point fraction and matrix language. See rnd-docs/06 section 3."""
    tokens = normalize_text(text).split()
    scripts = [token_script(t) for t in tokens]
    lang = [s for s in scripts if s != "neutral"]

    n_bn = lang.count("bn")
    n_en = lang.count("en")
    n = len(lang)

    if n == 0:
        cmi = 0.0
    else:
        cmi = 100.0 * (1 - max(n_bn, n_en) / n)

    switches = sum(1 for a, b in zip(lang, lang[1:]) if a != b)
    spf = switches / (n - 1) if n > 1 else 0.0

    return {
        "cmi": round(cmi, 2),
        "spf": round(spf, 3),
        "matrix_language": "bn" if n_bn >= n_en else "en",
        "n_tokens": len(tokens),
        "n_bn_tokens": n_bn,
        "n_en_tokens": n_en,
        "latin_ratio": round(n_en / n, 3) if n else 0.0,
    }


if __name__ == "__main__":
    from common import use_utf8_stdout

    use_utf8_stdout()

    samples = [
        "আমাদের startupটা basically একটা B2B SaaS platform।",
        "গত বছর আমাদের revenue ছিল ৫ কোটি টাকা।",
        "We are going to deploy this next week.",
    ]
    for s in samples:
        print(f"{s}\n  -> {normalize_text(s)}\n  -> {codemix_stats(s)}\n")
