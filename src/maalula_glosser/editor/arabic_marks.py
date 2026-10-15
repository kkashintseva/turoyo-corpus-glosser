"""The ᴬ…ᴬ marker that brackets an Arabic insert.

The transcription writes a switch into Arabic between two capital A's —
``Asənt əlxamsā u ʕəšrīnA``, ``AəlmasīḥīyeA``. In the published document they
are superscript, so they read as an annotation on the text rather than as
letters of it.

They are found in pairs, never singly. Maalula has ordinary words that begin
with a capital A — Amērca, Ana — and no rule about a single letter can tell
them from an opening marker. An opening marker with no closing marker after it
is therefore left alone: a missed pair costs nothing, and a wrong guess would
turn the A of Amērca into a diacritic.
"""
from __future__ import annotations

import re
import unicodedata

MARK = "A"

# `Asənt`, `AəlmasīḥīyeA`: the marker sits at a word edge against a letter, so a
# bare "A" standing as its own word is not one.
_WORD = re.compile(r"\S+")


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value or "")


def _letter(char: str) -> bool:
    return unicodedata.category(char).startswith("L")


def opens(word: str) -> bool:
    return len(word) > 1 and word[0] == MARK and _letter(word[1])


def _closing_offset(word: str) -> int | None:
    """Where the closing mark sits inside `word`, sentence punctuation aside.

    A span very often ends a clause — `Adaraže ʔūlāA.` — so the mark is not
    always the last character.
    """
    end = len(word)
    while end > 0 and unicodedata.category(word[end - 1]).startswith("P"):
        end -= 1
    if end > 1 and word[end - 1] == MARK and _letter(word[end - 2]):
        return end - 1
    return None


def closes(word: str) -> bool:
    return _closing_offset(word) is not None


def spans(text: str) -> list[tuple[int, int]]:
    """Offsets of the marker characters that make up complete ᴬ…ᴬ pairs.

    The nearest opener wins. `Awwal ma nčašraṯ AəlmasīḥīyeA` has two words
    beginning with a capital A, and only the second one is a marker; pairing the
    first with the closer would superscript the A of an ordinary word and swallow
    half the sentence. A later opener therefore cancels an earlier one.

    A line may open with an insert, so the first word counts like any other.
    What is left is the case of a sentence-initial `Ana` with an unpaired closer
    somewhere after it, which no rule about a single letter can catch — that is
    what marking a span by hand is for.
    """
    text = _nfc(text)
    words = [(m.start(), m.end(), m.group(0)) for m in _WORD.finditer(text)]
    found: list[tuple[int, int]] = []
    pending: int | None = None
    for position, (start, _end, word) in enumerate(words):
        closing = _closing_offset(word)
        # A standalone ``A.`` closes a span only when this same line already
        # has a pending opener. In isolation it remains ordinary text (and may
        # independently be a speaker label).
        if closing is None and pending is not None:
            end = len(word)
            while end > 0 and unicodedata.category(word[end - 1]).startswith("P"):
                end -= 1
            if end == 1 and word[0] == MARK:
                closing = 0
        # a one-word insert opens and closes on itself, but a bare "AA" is not a
        # span — there has to be a letter between the two marks
        if opens(word) and closing is not None and len(word) > 2:
            found.append((start, start + closing))
            pending = None
            continue
        if opens(word):
            pending = start
            continue
        if closing is not None and pending is not None:
            found.append((pending, start + closing))
            pending = None
    return found


def runs(text: str) -> list[tuple[str, bool]]:
    """`text` split into `(piece, superscript)`, adjacent pieces never merged.

    Round-trips: joining every piece gives the input back, so nothing is lost by
    rendering through this.
    """
    text = _nfc(text)
    marks = sorted({offset for pair in spans(text) for offset in pair})
    if not marks:
        return [(text, False)] if text else []
    out: list[tuple[str, bool]] = []
    cursor = 0
    for offset in marks:
        if offset > cursor:
            out.append((text[cursor:offset], False))
        out.append((text[offset], True))
        cursor = offset + 1
    if cursor < len(text):
        out.append((text[cursor:], False))
    return out


def runs_per_word(words: list[str]) -> list[list[tuple[str, bool]]]:
    """`runs` for a row of columns, decided over the whole row.

    A span usually runs across several words, so a column on its own cannot tell
    whether its leading A is a marker. The row is measured as one string and the
    answer handed back column by column.
    """
    joined = " ".join(_nfc(word) for word in words)
    marks = {offset for pair in spans(joined) for offset in pair}
    out: list[list[tuple[str, bool]]] = []
    cursor = 0
    for word in words:
        word = _nfc(word)
        pieces: list[tuple[str, bool]] = []
        plain = ""
        for index, char in enumerate(word):
            if cursor + index in marks:
                if plain:
                    pieces.append((plain, False))
                    plain = ""
                pieces.append((char, True))
            else:
                plain += char
        if plain:
            pieces.append((plain, False))
        out.append(pieces)
        cursor += len(word) + 1
    return out
