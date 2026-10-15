"""Conservative NC1/NC2 labelling for strong stem-I perfect forms.

The internal TAM remains PRF.  NC1/NC2 are display-level subclasses inferred
only when the linked lexeme and the written consonant skeleton are decisive.
Anything weak, derived, unlinked, unaligned, or otherwise irregular stays PRF.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping

CONS = frozenset("bcčdḏfgġhḥjklmnpqrsṣštṭṯvwxyzžʔʕḳ")
WEAK_RADICALS = frozenset("wyʔ")
_EMPHATIC_FOLD = str.maketrans("ḳṣṭḏṯẓ", "kstdtz")
_PRF = re.compile(r"\.PRF\b", re.IGNORECASE)


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value or "")


def consonants(value: str | Iterable[str]) -> str:
    if not isinstance(value, str):
        value = "".join(value)
    return "".join(ch for ch in _nfc(value) if ch in CONS)


def _fold(value: str) -> str:
    return value.translate(_EMPHATIC_FOLD)


def _regular_strong_root(value: str | Iterable[str]) -> str | None:
    root = consonants(value)
    folded = _fold(root)
    if len(root) != 3 or set(root) & WEAK_RADICALS:
        return None
    # Repeated-radical roots are outside the deliberately narrow first patch.
    if len(set(folded)) != 3:
        return None
    return root


def is_unambiguous_strong_stem_i(lexeme: Mapping[str, object] | None) -> bool:
    """True only for a regular strong lexeme whose citation licenses stem I.

    In the current lexicon a strong G perfect begins with the citation vowel
    ``i-`` and has exactly the linked root skeleton (``ikṭal``, ``inḥeč``).
    This rejects derived citations such as ``arxeb``, ``ḥassel`` and
    ``šammaʕ`` even when they share the same dictionary root.
    """
    if not lexeme:
        return False
    root = _regular_strong_root(str(lexeme.get("root_canonical") or ""))
    if root is None:
        return False
    values = [str(lexeme.get("lemma_canonical") or "")]
    values.extend(str(value) for value in lexeme.get("principal_parts") or ())
    target = _fold(root)
    for value in values:
        first_word = value.strip().split(maxsplit=1)[0] if value.strip() else ""
        for alternative in first_word.split("/"):
            citation = alternative.strip("(),;[] ")
            if citation.lower().startswith("i") \
                    and _fold(consonants(citation)) == target:
                return True
    return False


def extract_prf_stem(segmentation: str, gloss: str) -> str | None:
    """Return the segment aligned with the single lexical ``.PRF`` cell."""
    segmentation = _nfc(segmentation)
    gloss = gloss or ""
    if re.findall(r"[-=]", segmentation) != re.findall(r"[-=]", gloss):
        return None
    segments = re.split(r"[-=]", segmentation)
    glosses = re.split(r"[-=]", gloss)
    if len(segments) != len(glosses):
        return None
    indexes = [index for index, cell in enumerate(glosses) if _PRF.search(cell)]
    if len(indexes) != 1:
        return None
    return segments[indexes[0]].strip("() ") or None


def classify_prf_stem(root: str | Iterable[str], stem: str) -> str | None:
    """Classify an explicitly written stem skeleton, otherwise abstain."""
    root_cons = _regular_strong_root(root)
    if root_cons is None:
        return None
    c1, c2, c3 = _fold(root_cons)
    stem_cons = _fold(consonants(stem))
    if stem_cons == c1 + c2 + c3:
        return "NC1"
    if stem_cons == c1 + c2 + c2 + c3:
        return "NC2"
    return None


def classify_prf_analysis(
    lexeme: Mapping[str, object] | None,
    segmentation: str,
    gloss: str,
) -> tuple[str, str] | None:
    """Return ``(NC subclass, lexical stem)`` for a decisive A1 analysis."""
    if not _PRF.search(gloss or "") or not is_unambiguous_strong_stem_i(lexeme):
        return None
    stem = extract_prf_stem(segmentation, gloss)
    if stem is None:
        return None
    subclass = classify_prf_stem(
        str((lexeme or {}).get("root_canonical") or ""), stem)
    return (subclass, stem) if subclass else None


def classify_prf_profile(
    root: str | Iterable[str], stems: Iterable[str | None]
) -> str | None:
    """Infer one subclass only when every available profile stem agrees."""
    available = [stem for stem in stems if stem]
    if not available:
        return None
    labels = [classify_prf_stem(root, stem) for stem in available]
    if any(label is None for label in labels):
        return None
    unique = set(labels)
    return next(iter(unique)) if len(unique) == 1 else None


def replace_prf_label(gloss: str, subclass: str | None) -> str:
    """Replace only the lexical TAM label; aliases at ingestion stay intact."""
    if subclass not in {"NC1", "NC2"}:
        return gloss
    return _PRF.sub(f".{subclass}", gloss, count=1)
