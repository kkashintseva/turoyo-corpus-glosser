from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import unicodedata
from typing import Iterable, Iterator, Literal

DEFAULT_RITTER_PATH = (
    Path(__file__).resolve().parent
    / "resources"
    / "dictionaries"
    / "ritter.txt"
)

# Vowels may differ between the corpus transcription and Ritter. Consonants,
# including š/ž/ḥ/ṭ/ṣ/ḏ/ṯ/ġ/ǧ/č/ʕ/ʔ etc., are deliberately NOT folded.
#
# Ordinary a/e/i/o/u vowels are detected after Unicode decomposition, so
# ā, ă, á, â, ē, ī, ō, ū, ŭ, ắ, etc. are all treated as vowels without a
# destructive "strip all diacritics" operation.
_EXTRA_VOWELS = frozenset({"ə", "ı", "ɨ", "ǝ", "æ", "œ", "ø", "ɐ", "ɑ"})
_ASCII_VOWELS = frozenset("aeiou")

# The supplied revised Ritter file has the lexicon first and an editorial
# legend/bibliography afterwards. Stop before the legend so "ar arabisch",
# "bed beduinisch", etc. never become fake lexical entries.
_END_MARKER = "tiefgestellte Textangabe"

# Section headings inside the dictionary are bare letters such as "ʕ", "a",
# "b", ... and occasional page-number artefacts.
_SECTION_RE = re.compile(r"^[^\W\d_]{1,2}$", re.UNICODE)

# Alternatives written in a single headword token, e.g.
# zuwade/zuwwade or žēb/ǧēb.
_ALT_SPLIT_RE = re.compile(r"/+")

# Do not strip parentheses here: Ritter sometimes writes optional vowels as
# sūr(o), and leaving the source spelling intact is safer. Parentheses are
# ignored automatically by consonant_skeleton().
_STRIP_AROUND = " \t\r\n,;:.!?\"“”„'‘’`[]{}<>«»"


@dataclass(frozen=True, slots=True)
class RitterEntry:
    """One lexical line from the Ritter/Betzold dictionary source."""

    entry_id: str
    line_number: int
    headword: str
    lookup_forms: tuple[str, ...]
    raw_entry: str

    @property
    def skeleton(self) -> str:
        return consonant_skeleton(self.headword)


@dataclass(frozen=True, slots=True)
class RitterMatch:
    """A dictionary candidate returned for a corpus form."""

    kind: Literal["exact", "vowel_variant"]
    entry: RitterEntry
    matched_form: str
    skeleton: str
    vowel_distance: int

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "entry_id": self.entry.entry_id,
            "line_number": self.entry.line_number,
            "headword": self.entry.headword,
            "matched_form": self.matched_form,
            "skeleton": self.skeleton,
            "vowel_distance": self.vowel_distance,
            "raw_entry": self.entry.raw_entry,
        }


def _clusters(text: str) -> Iterator[str]:
    """Yield Unicode base+combining-mark clusters without external deps."""
    current = ""
    for char in unicodedata.normalize("NFD", text):
        if unicodedata.combining(char):
            current += char
            continue
        if current:
            yield current
        current = char
    if current:
        yield current


def _normalize_dashes(text: str) -> str:
    return (
        text.replace("–", "-")
        .replace("—", "-")
        .replace("‑", "-")
        .replace("−", "-")
    )


def _clean_source_form(text: str) -> str:
    """Clean edge punctuation while preserving Ritter's displayed spelling."""
    value = unicodedata.normalize("NFC", text or "").strip()
    return _normalize_dashes(value).strip(_STRIP_AROUND)


def normalize_lookup_form(text: str) -> str:
    """Conservative normalization for exact dictionary lookup.

    This intentionally does *not* strip consonantal diacritics. It only:
    - NFC-normalizes Unicode;
    - case-folds;
    - trims surrounding punctuation/whitespace;
    - normalizes common dash glyphs to ASCII "-".
    """
    return _clean_source_form(text).casefold()


def _is_vowel_cluster(cluster: str) -> bool:
    base = cluster[0].casefold()
    return base in _ASCII_VOWELS or base in _EXTRA_VOWELS


def consonant_skeleton(text: str) -> str:
    """Return the exact consonant sequence, ignoring vowels only.

    Examples:
        qəbəl   -> qbl
        qabila  -> qbl
        šuro    -> šr
        suro    -> sr        (NOT the same skeleton)

    Consonant identity, order and multiplicity are all preserved.
    Punctuation, morph-boundary punctuation, spaces and digits are ignored.
    """
    value = normalize_lookup_form(text)
    out: list[str] = []

    for cluster in _clusters(value):
        base = cluster[0]
        if _is_vowel_cluster(cluster):
            continue
        category = unicodedata.category(base)
        if category.startswith("L"):
            out.append(unicodedata.normalize("NFC", cluster))

    return "".join(out)


def vowel_signature(text: str) -> str:
    """Return a reduced vowel sequence for ranking same-skeleton candidates.

    This never filters candidates; it only gives more similar vocalism a
    better rank. Ordinary vowel diacritics collapse to a/e/i/o/u while
    schwa-like letters remain distinct.
    """
    value = normalize_lookup_form(text)
    out: list[str] = []

    for cluster in _clusters(value):
        if not _is_vowel_cluster(cluster):
            continue
        base = cluster[0].casefold()
        if base in _ASCII_VOWELS:
            out.append(base)
        elif base == "ǝ":
            out.append("ə")
        else:
            out.append(base)

    return "".join(out)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current
    return previous[-1]


def _headword_forms(first_token: str) -> tuple[str, ...]:
    """Extract safe headword alternatives from the first dictionary token."""
    forms: list[str] = []
    seen: set[str] = set()

    for piece in _ALT_SPLIT_RE.split(first_token):
        form = _clean_source_form(piece)
        normalized = normalize_lookup_form(form)
        if not form or not normalized:
            continue
        if not any(unicodedata.category(ch).startswith("L") for ch in form):
            continue
        if normalized not in seen:
            seen.add(normalized)
            forms.append(form)

    return tuple(forms)


def parse_ritter_text(text: str) -> list[RitterEntry]:
    """Parse the supplied Ritter/Betzold TXT conservatively.

    The source is one lexical entry per physical line. We deliberately do
    not try to parse German senses, etymologies, roots or corpus references:
    those fields are heterogeneous and the raw source line is retained
    losslessly for display. For word lookup we only need the leading Turoyo
    headword and safe slash-separated alternatives.
    """
    entries: list[RitterEntry] = []
    started = False

    for line_number, original in enumerate(text.splitlines(), start=1):
        raw = original.strip()
        if not raw:
            continue

        if not started:
            if raw == "ʕ":
                started = True
            continue

        if raw.startswith(_END_MARKER):
            break

        if raw.isdigit() or (_SECTION_RE.fullmatch(raw) and len(raw) <= 2):
            continue

        first_token = raw.split(maxsplit=1)[0].strip()
        forms = _headword_forms(first_token)
        if not forms:
            continue

        headword = forms[0]
        entries.append(
            RitterEntry(
                entry_id=f"ritter:{line_number}",
                line_number=line_number,
                headword=headword,
                lookup_forms=forms,
                raw_entry=raw,
            )
        )

    if not entries:
        raise ValueError(
            "No Ritter dictionary entries found. "
            "Expected the supplied Ritter/Betzold TXT format."
        )

    return entries


class RitterDictionary:
    """In-memory exact + consonant-skeleton index for Ritter."""

    def __init__(self, entries: Iterable[RitterEntry], *, source: Path | None = None):
        self.source = source
        self.entries = tuple(entries)

        exact: dict[str, list[tuple[RitterEntry, str]]] = {}
        skeletons: dict[str, list[tuple[RitterEntry, str]]] = {}

        for entry in self.entries:
            for form in entry.lookup_forms:
                normalized = normalize_lookup_form(form)
                if not normalized:
                    continue
                exact.setdefault(normalized, []).append((entry, form))

                skeleton = consonant_skeleton(form)
                if skeleton:
                    skeletons.setdefault(skeleton, []).append((entry, form))

        self._exact = exact
        self._skeletons = skeletons

    @classmethod
    def from_file(cls, path: Path | str) -> "RitterDictionary":
        source = Path(path).expanduser().resolve()
        text = source.read_text(encoding="utf-8-sig")
        return cls(parse_ritter_text(text), source=source)

    @classmethod
    def default(cls) -> "RitterDictionary":
        return cls.from_file(DEFAULT_RITTER_PATH)

    def lookup(
        self,
        query: str,
        *,
        limit: int = 12,
        include_vowel_variants: bool = True,
        min_skeleton_consonants: int = 2,
    ) -> list[RitterMatch]:
        """Look up a Turoyo form.

        Ranking:
        1. exact orthographic matches;
        2. same consonant skeleton, ranked only by vocalic distance.

        No candidate with a different consonant skeleton is ever returned as a
        vowel variant. For one-consonant words, skeleton expansion is disabled
        by default because it is usually too ambiguous.
        """
        if limit <= 0:
            return []

        normalized = normalize_lookup_form(query)
        if not normalized:
            return []

        query_skeleton = consonant_skeleton(normalized)
        query_vowels = vowel_signature(normalized)

        matches: list[RitterMatch] = []
        seen: set[tuple[str, str, str]] = set()

        def add(
            kind: Literal["exact", "vowel_variant"],
            entry: RitterEntry,
            form: str,
        ) -> None:
            key = (kind, entry.entry_id, normalize_lookup_form(form))
            if key in seen:
                return
            seen.add(key)
            matches.append(
                RitterMatch(
                    kind=kind,
                    entry=entry,
                    matched_form=form,
                    skeleton=consonant_skeleton(form),
                    vowel_distance=_levenshtein(
                        query_vowels,
                        vowel_signature(form),
                    ),
                )
            )

        for entry, form in self._exact.get(normalized, ()):
            add("exact", entry, form)

        if (
            include_vowel_variants
            and query_skeleton
            and len(query_skeleton) >= min_skeleton_consonants
        ):
            exact_pairs = {
                (match.entry.entry_id, normalize_lookup_form(match.matched_form))
                for match in matches
                if match.kind == "exact"
            }
            variants: list[RitterMatch] = []

            for entry, form in self._skeletons.get(query_skeleton, ()):
                pair = (entry.entry_id, normalize_lookup_form(form))
                if pair in exact_pairs:
                    continue
                variants.append(
                    RitterMatch(
                        kind="vowel_variant",
                        entry=entry,
                        matched_form=form,
                        skeleton=query_skeleton,
                        vowel_distance=_levenshtein(
                            query_vowels,
                            vowel_signature(form),
                        ),
                    )
                )

            variants.sort(
                key=lambda match: (
                    match.vowel_distance,
                    abs(
                        len(normalize_lookup_form(match.matched_form))
                        - len(normalized)
                    ),
                    match.entry.line_number,
                    normalize_lookup_form(match.matched_form),
                )
            )
            for match in variants:
                add(match.kind, match.entry, match.matched_form)

        matches.sort(
            key=lambda match: (
                0 if match.kind == "exact" else 1,
                match.vowel_distance,
                abs(
                    len(normalize_lookup_form(match.matched_form))
                    - len(normalized)
                ),
                match.entry.line_number,
                normalize_lookup_form(match.matched_form),
            )
        )
        return matches[:limit]

    def status(self) -> dict[str, object]:
        ambiguous_skeletons = sum(
            1
            for rows in self._skeletons.values()
            if len({entry.entry_id for entry, _ in rows}) > 1
        )
        return {
            "source": str(self.source) if self.source else None,
            "entries": len(self.entries),
            "lookup_forms": sum(len(entry.lookup_forms) for entry in self.entries),
            "exact_forms": len(self._exact),
            "consonant_skeletons": len(self._skeletons),
            "ambiguous_skeletons": ambiguous_skeletons,
        }


@lru_cache(maxsize=1)
def default_ritter_dictionary() -> RitterDictionary:
    """Load the bundled Ritter dictionary once per Python process."""
    return RitterDictionary.default()
