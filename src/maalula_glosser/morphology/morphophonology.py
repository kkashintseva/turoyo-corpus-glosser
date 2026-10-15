"""Declarative, class-scoped morphophonology for the v4 generator.

Every rule records before/after in a trace. Nothing here is global: a rule
fires only when the caller applies it for a licensed class/boundary. The
inserted epenthetic vowel is never a morpheme.
"""
from __future__ import annotations

import unicodedata

# voiceless consonants that trigger b→p on a preceding stem-final b
VOICELESS = set("ptkčṭḳṣšfxẖḥ")


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value or "")


def step(rule: str, before: str, after: str) -> dict:
    return {"rule": rule, "before": before, "after": after}


def attach_construct_l(stem: str) -> tuple[str, list[dict]]:
    """Underlying =l after a noun stem; insert epenthetic i on a hard cluster.

    tarb + =l -> tarb=il (i epenthetic, not a morpheme); ešn + =l -> ešn=l.
    Epenthesis is licensed when the stem ends in two consonants.
    """
    trace = []
    base = stem
    if _ends_in_cluster(stem):
        surface = f"{stem}=il"
        trace.append(step("EPENTHESIS_I", f"{base}+l", f"{stem}=il"))
    else:
        surface = f"{stem}=l"
        trace.append(step("ATTACH_CSTR_L", f"{base}+l", surface))
    return surface, trace


def devoice_b_before_voiceless(stem: str, clitic: str) -> tuple[str, list[dict]]:
    """b -> p before a voiceless-initial enclitic: tarb + =xun -> tarp=xun."""
    trace = []
    if stem.endswith("b") and clitic[:1] in VOICELESS:
        new = stem[:-1] + "p"
        trace.append(step("DEVOICE_B_TO_P", f"{stem}={clitic}", f"{new}={clitic}"))
        return new, trace
    return stem, trace


def assimilate_n_l(prefix_n: str, host: str) -> tuple[str, list[dict]]:
    """n + l -> ll at a licensed boundary."""
    if prefix_n.endswith("n") and host.startswith("l"):
        merged = prefix_n[:-1] + "l" + host
        return merged, [step("N_L_ASSIMILATION", f"{prefix_n}+{host}", merged)]
    return prefix_n + host, []


def shorten_ee_to_i(stem: str) -> tuple[str, list[dict]]:
    """ē -> i under a licensed stress shift (class-scoped, caller decides)."""
    if "ē" in stem:
        new = stem.replace("ē", "i", 1)
        return new, [step("SHORTEN_EE_TO_I", stem, new)]
    return stem, []


def shorten_oo_to_u(stem: str) -> tuple[str, list[dict]]:
    if "ō" in stem:
        new = stem.replace("ō", "u", 1)
        return new, [step("SHORTEN_OO_TO_U", stem, new)]
    return stem, []


def historical_oo_to_a(stem: str) -> tuple[str, list[dict]]:
    """ō (< *ā) -> a; only for lexemes flagged with this class."""
    if "ō" in stem:
        new = stem.replace("ō", "a", 1)
        return new, [step("HISTORICAL_OO_TO_A", stem, new)]
    return stem, []


_VOWELS = set("aeiouāēīōūəAEIOUĀĒĪŌŪƏ")


def simplify_geminate_before_consonant(text: str) -> str:
    """CxCx + C -> Cx C: a geminate before a consonant is written single.

    The author (01.10.2026): yšattrun "usually simplifies to" yšatrun. The
    texts agree for plosives too: yišatrell (glossed_all), mzapnin
    (glossed_all, Muše Sanžar), ču=mbakr-in, ni-mfark-il=le, šakl-ul-l=l
    (Shabo), šaklaṯ; beside them the full spellings survive (mšakkla,
    taššrann, čġayyraṯ), so a reader accepts both and a generator keeps
    Arnold's (yšattrun, S. 80). Separators stay: a geminate split by one is
    not a stem geminate (the linker ḥm-ič-č=l is written ičč-l).
    """
    out: list[str] = []
    for index, char in enumerate(text):
        if out and char == out[-1] and char.isalpha() and char not in _VOWELS:
            following = index + 1
            while following < len(text) and text[following] in "-=":
                following += 1
            if following < len(text) and text[following].isalpha() \
                    and text[following] not in _VOWELS:
                continue
        out.append(char)
    return "".join(out)


def _after_consonant(written: list[str] | str) -> bool:
    """Whether the next letter would stand directly after a consonant of the
    same morpheme (a separator, a vowel or the word edge before it: no)."""
    index = len(written) - 1
    while index >= 0 and unicodedata.combining(written[index]):
        index -= 1                      # the dot below of ḏ̣ belongs to its ḏ
    return index >= 0 and written[index].isalpha() \
        and written[index] not in _VOWELS


def voice_p_after_consonant(text: str) -> str:
    """Lookup spelling of a word: p after a consonant read as b.

    A root b standing after a consonant is written p in the texts as often as
    not (ṭalpilla, ṭalpōla for √ṭlb beside ṭalbilla; xaṯpilla, xaṯpunnlə for
    √xṯb; arxp=a; Arnold himself cites aḏpel, yaspeʕ, yaṣpar under ḏbl, sbʕ,
    ṣbr), while the stems of the lexicon keep the b of the root. Only for
    finding a stem: the word is never rewritten, and `respell_devoiced_b`
    puts the written p back into line 2. A geminate pp is left alone (ʕappar
    is stored with its pp).
    """
    out: list[str] = []
    for char in nfc(text):
        if char == "p" and out and out[-1] != "p" and _after_consonant(out):
            out.append("b")
        else:
            out.append(char)
    return "".join(out)


def devoice_b_after_consonant(text: str) -> str:
    """The same stem as the texts may write it: b after a consonant as p
    (ṭalb -> ṭalp). The inverse of `voice_p_after_consonant`; a geminate bb
    is not touched."""
    out: list[str] = []
    for char in nfc(text):
        if char == "b" and out and out[-1] != "b" and _after_consonant(out):
            out.append("p")
        else:
            out.append(char)
    return "".join(out)


def respell_devoiced_b(segmentation: str, written: str) -> str | None:
    """A generated segmentation spelled as the word is written, where the two
    differ only in a b the text writes p after a consonant of the same
    morpheme (ṭalb-il=la read off ṭalpilla is ṭalp-il=la). The segmentation
    itself if nothing differs; None if anything else does.
    """
    written = nfc(written).replace("-", "").replace("=", "")
    out: list[str] = []
    position = 0
    for char in nfc(segmentation):
        if char in "-=":
            out.append(char)
            continue
        if position >= len(written):
            return None
        letter = written[position]
        position += 1
        if char == letter:
            out.append(char)
        elif char == "b" and letter == "p" and out[-1:] != ["b"] \
                and _after_consonant(out):
            out.append("p")
        else:
            return None
    return "".join(out) if position == len(written) else None


def _ends_in_cluster(stem: str) -> bool:
    cons = "bcčdḏfgġhḥjklmnpqrsṣštṭṯvwxyzžʔʕḳ"
    return len(stem) >= 2 and stem[-1] in cons and stem[-2] in cons
