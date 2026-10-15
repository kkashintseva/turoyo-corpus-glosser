"""Gloss running Maalula text: token analysis + sentence-level IGT.

External sandhi (``‿``) is preserved in the original line; the segmentation
line shows the isolated forms it joins.
"""

from __future__ import annotations

import copy
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

from maalula_glosser.bound_labels import typed_bound_options

from maalula_glosser.compositional import RULES, CompositionalAnalyzer, nfc
from maalula_glosser.normalization.gloss import (iv_person_prefix,
                                                  nc_plural_participle,
                                                  render_authoritative_gloss,
                                                  sc_plural_person)
from maalula_glosser.normalization.transcription import fold_allophones
from maalula_glosser.reusable import morphology_surfaces, surface_key

_SENTENCE = RULES.get("sentence_level", {})

UNDERTIE = "‿"
_TOKEN_RE = re.compile(r"[^\s]+")
# Everything Unicode calls punctuation is stripped from the edges of a token,
# except the three characters that carry meaning here: the sandhi tie and the
# two segmentation separators. A hard-coded list missed "…" and left ʕam… —
# a perfectly analysable ʕa(m)= PP= — unrecognised.
_KEEP = {"-", "=", UNDERTIE}


def _strip_punctuation(token: str) -> str:
    def is_punct(char: str) -> bool:
        return char not in _KEEP and unicodedata.category(char).startswith("P")

    start, end = 0, len(token)
    while start < end and is_punct(token[start]):
        start += 1
    while end > start and is_punct(token[end - 1]):
        end -= 1
    return token[start:end]


# A capital at one of these written boundaries is orthographic first and
# lexical only if the lower-case analysis fails.  Includes the ASCII and
# Unicode quote/ellipsis characters actually found in the transcripts.
_CAPITAL_BOUNDARIES = frozenset('.?!:…"“”„«»‘’')


def _contextual_capital(sentence: str, token: dict[str, Any],
                        piece_index: int) -> bool:
    """Whether initial case is plausibly imposed by punctuation/position."""
    if piece_index != 0:
        return False
    core = token.get("core") or ""
    if not core[:1].isupper():
        return False
    start = token.get("start")
    if start is None:
        return False
    if start == 0:
        return True

    raw = token.get("original") or ""
    first_letter = next((i for i, char in enumerate(raw) if char.isalpha()), None)
    if first_letter is not None and any(
            char in _CAPITAL_BOUNDARIES for char in raw[:first_letter]):
        return True

    left = sentence[:start].rstrip()
    if not left:
        return True
    if left.endswith("..."):
        return True
    return left[-1] in _CAPITAL_BOUNDARIES


def split_sentences(text: str) -> list[str]:
    source = nfc(text).strip()
    if not source:
        return []
    parts: list[str] = []
    start = 0
    for match in re.finditer(r"[.!?]\s+", source):
        end = match.start() + 1
        token = source[start:end].rsplit(None, 1)[-1]
        letters = token[:-1] if token.endswith(".") else ""
        dotted_speaker = (
            token not in {"A.", "E."}
            and 1 <= len(letters) <= 2
            and all(ch.isalpha() and ch.isupper() for ch in letters)
        )
        if dotted_speaker:
            continue
        parts.append(source[start:end].strip())
        start = match.end()
    tail = source[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def tokenize(sentence: str) -> list[dict[str, Any]]:
    """Whitespace tokens; a sandhi-joined token is split into isolated parts."""
    tokens = []
    for match in _TOKEN_RE.finditer(sentence):
        raw = match.group(0)
        core = _strip_punctuation(raw)
        if not core:
            continue
        # Punctuation can sit *inside* a sandhi group (yaskull…‿l-ṭuflō), so
        # each piece is cleaned again after the split, not only the whole token.
        pieces = [p for p in (_strip_punctuation(piece)
                              for piece in core.split(UNDERTIE)) if p]
        if not pieces:
            continue
        tokens.append({"original": raw, "core": core,
                       "pieces": pieces, "sandhi": UNDERTIE in core,
                       "start": match.start(), "end": match.end()})
    return tokens


PROGRESSIVE_HOSTS = set(_SENTENCE.get("progressive_hosts", ["ʕa", "ʕam", "ʕamma"]))


def _is_present_verb(record: dict[str, Any] | None) -> bool | None:
    """True only when the neighbour is an NC/PRS verb.

    Three outcomes, not two.  ``PP=`` is the progressive marker, so NC1/NC2
    (PRF), SC/PST, PC/SBJV and IPV are no evidence *for* it -- but they are a
    verb, and a preposition cannot govern a verbal stem either, so neither
    reading wins and the token keeps both with its review flag (None).  Only a
    nominal host decides for the ordinary preposition (False).
    """
    if not record or not record.get("gloss"):
        return None
    features = record.get("features") or {}
    tam = str(features.get("tam") or "").upper()
    if tam:
        return True if tam == "PRS" else None
    gloss = record["gloss"]
    if re.search(r"(?:^|[.=\-])(?:NC|PRS)(?=$|[.=\-])", gloss):
        return True
    if re.search(
            r"(?:^|[.=\-])(?:NC1|NC2|PRF|SC|PST|PC|SBJV|IPV)"
            r"(?=$|[.=\-])", gloss):
        return None
    if any(tag in gloss for tag in ("-FREE", "village", "pn", "PN")):
        return False
    return None


def disambiguate_progressive(analyses: list[dict[str, Any]]) -> None:
    """Choose between ʕa(m)= PP= and the plain preposition ʕa, in place.

    Narrow rule, deliberately not a model of the sentence:
      ʕa(m) + NC/PRS verb      → progressive PP=
      ʕa    + nominal host     → ordinary preposition
      ʕa    + any other verb   → undecided, both readings stay
    With no evidence either way both readings stay and the token keeps its
    review flag.
    """
    for index, record in enumerate(analyses):
        if nfc(record.get("normalized") or "").lower() not in PROGRESSIVE_HOSTS:
            continue
        options = [{"segmentation": record["segmentation"], "gloss": record["gloss"],
                    "tier": record.get("tier"), "lemma": record.get("lemma")}]
        options += list(record.get("alternatives") or [])
        progressive = [o for o in options if (o["gloss"] or "").startswith("PP")]
        prepositional = [o for o in options if not (o["gloss"] or "").startswith("PP")]
        if not progressive or not prepositional:
            continue
        host = analyses[index + 1] if index + 1 < len(analyses) else None
        # A bare final ʕa really is undecidable.  Once a host exists, however,
        # PP is positively licensed only by an NC/PRS analysis of that host.
        # Unknown hosts and every other TAM therefore cannot keep PP merely
        # because PP happened to win the token-local score for bare ʕa.
        if host is None:
            record["uncertainty"] = list(record.get("uncertainty") or []) + [
                "ʕa: progressive vs preposition undecided, both kept"]
            continue
        verdict = _is_present_verb(host)
        chosen = (progressive if verdict is True else prepositional)[0]
        # If NC/PRS is not positively established, PP is not even a licensed
        # alternative in this context.  Keep the competing prepositional
        # readings, but drop every PP reading before attach_proclitics() folds
        # the two written tokens into one word.
        rest = ([o for o in options if o is not chosen] if verdict is True else
                [o for o in prepositional if o is not chosen])
        record["segmentation"], record["gloss"] = chosen["segmentation"], chosen["gloss"]
        record["tier"], record["alternatives"] = chosen.get("tier"), rest
        record["context_rule"] = (
            "progressive: next token is an NC/PRS verb" if verdict is True else
            "preposition: PP rejected without an NC/PRS host")
        if verdict is None:
            record["uncertainty"] = list(record.get("uncertainty") or []) + [
                "ʕa: PP rejected because the next token is not confirmed NC/PRS"]


# Author-confirmed constructions and multiword expressions, from
# data/grammar_kb/rules_v3.json. Each promotes a reading inside one specific
# frame and leaves every other use of the same word alone.
CONSTRUCTIONS = _SENTENCE.get("constructions", [])
MWE = [{**entry, "tokens": tuple(entry["tokens"]), "parts": tuple(entry["parts"])}
       for entry in _SENTENCE.get("mwe", [])]


def apply_constructions(analyses: list[dict[str, Any]]) -> None:
    """Promote a reading only inside its confirmed frame, in place."""
    for index, record in enumerate(analyses):
        surface = nfc(record.get("normalized") or "").lower()
        for rule in CONSTRUCTIONS:
            if surface != rule["trigger"] or index + 1 >= len(analyses):
                continue
            nxt = analyses[index + 1]
            if nfc(nxt.get("normalized") or "").lower() != rule["target"]:
                continue
            options = [{"segmentation": nxt["segmentation"], "gloss": nxt["gloss"],
                        "tier": nxt.get("tier"), "lemma": nxt.get("lemma")}]
            options += list(nxt.get("alternatives") or [])
            if nxt["gloss"] != rule["gloss"]:
                nxt["alternatives"] = options
                nxt["gloss"] = rule["gloss"]
                nxt["construction"] = rule["id"]
                nxt["uncertainty"] = list(nxt.get("uncertainty") or []) + [rule["note"]]


def group_mwe(analyses: list[dict[str, Any]]) -> None:
    """Tag the members of a confirmed multiword expression with one id."""
    for entry in MWE:
        width = len(entry["tokens"])
        for index in range(len(analyses) - width + 1):
            window = analyses[index:index + width]
            if tuple(nfc(a.get("normalized") or "").lower() for a in window) != entry["tokens"]:
                continue
            for position, record in enumerate(window):
                record["mwe_id"] = entry["id"]
                record["mwe_gloss"] = entry["gloss"]
                record["gloss"] = entry["parts"][position]


# ---------------------------------------------------------------------------
# a proclitic and its host are one gloss
# ---------------------------------------------------------------------------
CONFIDENCE_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

# `-in`/`-an` and their spelled-out labels: what makes a third person plural,
# and therefore `3` rather than `3M`.
_PLURAL_TAIL = re.compile(r"[-=](in|an|ōn|ēn|un|3?[MF]P)\b")


def third_person(label: str, form: str) -> str:
    """`3M` on its own, `3` once something in the word makes it plural."""
    if label not in ("3", "3M"):
        return label
    return "3" if _PLURAL_TAIL.search(form) else "3M"

# Distinct forms remembered per glosser. The largest corpus text has under four
# thousand, so a whole session fits and the cap only guards a runaway.
TOKEN_CACHE = 20000

# How many further v4 readings of a word are kept on offer beside the first.
V4_VARIANTS = 6


def _host_source(host: dict[str, Any]) -> str:
    return host.get("source_token") or host.get("token") or ""


def is_bare_proclitic(record: dict[str, Any], proclitics: frozenset[str]) -> bool:
    """True for a registered proclitic standing on its own, host still to come.

    Bare is the whole point. ``lī=l to=1S`` and ``minn from.1S`` are the same
    prepositions with a pronoun already attached: joining them to the next word
    would gloss "to me house" as "to=house". A proclitic that still has its
    ``=`` empty is the only one looking for a host.
    """
    # M:/M. is a speaker tag in the transcripts, not the proclitic m-.
    # Keep it separate so attach_proclitics() cannot swallow the following word
    # before the editor marks the tag as #speaker.
    if nfc(record.get("source_token") or "").strip() in {"M:", "M."}:
        return False
    form = nfc(record.get("normalized") or "").lower()
    if form not in proclitics:
        return False
    segmentation = (record.get("segmentation") or "").rstrip("=")
    gloss = (record.get("gloss") or "").rstrip("=")
    if not segmentation or not gloss:
        return False
    return not any(mark in segmentation for mark in "-=") and \
        not any(mark in gloss for mark in "-=.")


def attach_proclitics(analyses: list[dict[str, Any]],
                      proclitics: frozenset[str],
                      analyse=None) -> list[dict[str, Any]]:
    """Fold each bare proclitic into the token that follows it.

    Maalula writes ``b Maʕlūla`` and ``b=Maʕlūla`` for the same thing, and the
    glossing standard prints one column either way: ``b=Maʕlūla / LOC=Maaloula``.
    Splitting on white space alone produced two columns, and — worse — a
    proclitic analysed on its own carries a dangling ``=`` that became a third
    column with nothing in it.

    Runs after the contextual passes, which need to see the two tokens apart:
    the progressive is decided by looking at the *next* token, so folding first
    would hide the evidence from it.
    """
    out: list[dict[str, Any]] = []
    index = 0
    while index < len(analyses):
        record = analyses[index]
        host = analyses[index + 1] if index + 1 < len(analyses) else None
        if host is None or not is_bare_proclitic(record, proclitics):
            out.append(record)
            index += 1
            continue
        same_group = (record.get("from_sandhi") and host.get("from_sandhi")
                      and record.get("source_token") == host.get("source_token"))
        gap = host.get("source_gap_before")
        # A punctuation-only token is absent from `analyses`, but it is still
        # a real boundary in the transcription. Do not turn
        # `m ... ġarība` into the fictitious single token `m ġarība`.
        if not same_group and gap is not None and not gap.isspace():
            out.append(record)
            index += 1
            continue
        out.append(_fold(record, host, analyse))
        index += 2
    return out


def _combined_alternatives(head_seg: str, head_gloss: str,
                           clitic: dict[str, Any], host: dict[str, Any],
                           chosen: tuple[str, str]) -> list[dict[str, Any]]:
    """Every reading the fold passed over, written out as whole words.

    Folding must not delete a reading. `ʕa` is either the preposition or the
    progressive PP=, and choosing between them is exactly what the hover tooltip
    is for, so both sides keep their competitors: the clitic's readings against
    the host's best, and the clitic's best against the host's.
    """
    heads = [(head_seg, head_gloss)] + [
        ((alternative.get("segmentation") or "").rstrip("="),
         (alternative.get("gloss") or "").rstrip("="))
        for alternative in clitic.get("alternatives") or []]
    hosts = [(host.get("segmentation") or host.get("normalized") or "",
              host.get("gloss") or "", host.get("tier"), host.get("lemma"))] + [
        (alternative.get("segmentation") or "", alternative.get("gloss") or "",
         alternative.get("tier"), alternative.get("lemma"))
        for alternative in host.get("alternatives") or []]
    pairs = [(head, hosts[0]) for head in heads[1:]] + \
            [(heads[0], body) for body in hosts[1:]]
    out, seen = [], {chosen}
    for head, body in pairs:
        reading = (f"{head[0]}={body[0]}", f"{head[1]}={body[1]}")
        if reading in seen:
            continue
        seen.add(reading)
        out.append({"segmentation": reading[0], "gloss": reading[1],
                    "tier": body[2], "lemma": body[3]})
    return out[:8]


def _fold(clitic: dict[str, Any], host: dict[str, Any],
          analyse=None) -> dict[str, Any]:
    """One record: the host's analysis with the clitic written in front of it.

    Ask about the whole form first. `b Maʕlūla` and `b=Maʕlūla` are the same
    word, and the corpus knows the second one: it glosses it `LOC=Maaloula`,
    where gluing the two halves separately would say `INS=Maaloula`, INS being
    what the bare preposition falls back to. Only when the joined form is
    unknown are the two analyses composed by hand — and either way the readings
    that lost are kept as alternatives.
    """
    same_group = (clitic.get("from_sandhi") and host.get("from_sandhi")
                  and clitic.get("source_token") == host.get("source_token"))
    joint = UNDERTIE if same_group else " "
    head_seg = (clitic.get("segmentation") or "").rstrip("=")
    head_gloss = (clitic.get("gloss") or "").rstrip("=")

    whole = None
    if analyse is not None:
        # the clitic's own segmentation, not the surface: a sentence-initial
        # `B̰` is normalised to `B` and segmented `b`, and it is the `b` that has
        # to match the head of the joined analysis
        candidate = analyse(f"{head_seg}={host.get('token') or ''}")
        segmentation = candidate.get("segmentation") or ""
        candidate_gloss = candidate.get("gloss") or ""
        candidate_head_gloss = candidate_gloss.split("=", 1)[0]
        # and only if it cuts where the writing cuts. `ʕal‿ōḏ` is also
        # analysable as `ʕa=lōḏ`, which is a different claim about the same
        # letters; the tie the author typed says the boundary is after `ʕal`.
        confirmed = any((e or {}).get("kind", "").startswith("reusable_morphology")
                        for e in candidate.get("evidence") or [] if isinstance(e, dict))
        empty_part = any(not part.strip("?") for part in re.split(r"[-=]", candidate_gloss))
        # a confirmed host is never traded for an unconfirmed whole reading
        # (nor for one a sentence rule has just decided, e.g. a restored =l)
        host_confirmed = (host.get("confirmed_span") or _confirmed(host)
                          or host.get("context_rule"))
        if segmentation.startswith(f"{head_seg}=") \
                and "=" in candidate_gloss and not empty_part \
                and (confirmed or not host_confirmed) \
                and (confirmed
                     or not (candidate_head_gloss == "PP" and head_gloss != "PP")):
            # An author-confirmed whole reading (`ʕa Lubnān` "PP=Lebanon")
            # is taken as confirmed; a composed one with an empty half
            # (`in=` from typed boundaries around an unknown host) is not a
            # reading of the whole at all.
            whole = candidate

    composed = (f"{head_seg}={host.get('segmentation') or host.get('normalized') or ''}",
                # an unanalysed host leaves the label empty rather than inventing
                # one; the cell then shows ?? beside a resolved clitic
                f"{head_gloss}={host.get('gloss') or ''}")
    alternatives = _combined_alternatives(head_seg, head_gloss, clitic, host,
                                          composed)

    merged = dict(host)
    if whole:
        for field in ("segmentation", "gloss", "lemma", "lemma_id",
                      "lexical_gloss", "tier", "confidence", "evidence", "trace"):
            merged[field] = whole.get(field)
        merged["normalized"] = whole.get("normalized") or ""
        # the hand-composed reading is a real competitor once the lookup wins
        alternatives = [{"segmentation": composed[0], "gloss": composed[1],
                         "tier": host.get("tier"), "lemma": host.get("lemma")},
                        *(a for a in alternatives
                          if (a["segmentation"], a["gloss"]) !=
                          (whole.get("segmentation"), whole.get("gloss")))]
    else:
        merged["segmentation"], merged["gloss"] = composed
        merged["normalized"] = (f"{clitic.get('normalized') or ''}"
                                f"{host.get('normalized') or ''}")
        merged["confidence"] = min(
            (clitic.get("confidence") or "LOW", host.get("confidence") or "LOW"),
            key=lambda level: CONFIDENCE_RANK.get(level, 0))

    merged["token"] = f"{clitic.get('token') or ''}{joint}{host.get('token') or ''}"
    if not same_group:
        source_gap = host.get("source_gap_before")
        source_joint = (source_gap if isinstance(source_gap, str)
                        and source_gap.isspace() else " ")
        merged["source_token"] = (f"{clitic.get('source_token') or ''}"
                                  f"{source_joint}{_host_source(host)}")
    merged["from_sandhi"] = bool(clitic.get("from_sandhi") or host.get("from_sandhi"))
    if clitic.get("piece_start") is not None:
        merged["piece_start"] = clitic["piece_start"]
    merged["proclitic"] = head_seg
    merged["alternatives"] = alternatives[:8]
    merged["uncertainty"] = list(clitic.get("uncertainty") or []) + \
        list(host.get("uncertainty") or [])
    merged["needs_review"] = bool(clitic.get("needs_review") or host.get("needs_review"))
    return merged


def _confirmed(record: dict[str, Any] | None) -> bool:
    """Whether a token reading comes from the author-confirmed journal."""
    return bool(record) and any(
        isinstance(e, dict) and str(e.get("kind") or "").startswith("reusable_morphology")
        for e in record.get("evidence") or [])


def _span_can_continue(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Two neighbouring records may be read as one written phrase."""
    if (left.get("from_sandhi") and right.get("from_sandhi")
            and left.get("source_token") == right.get("source_token")
            and right.get("source_gap_before") == ""):
        return True                       # two pieces of one sandhi word
    return _name_run_can_continue(left, right)


def apply_confirmed_spans(analyses: list[dict[str, Any]], has_reading,
                          analyse, max_width: int = 4,
                          proclitics: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    """Read 2-4 neighbouring words as the one unit the author confirmed.

    The journal holds whole-phrase readings typed in the editor --
    ``Sarces Šalhub̥`` "Sarkes Shalhub", ``b Lubnān`` "in=Lebanon",
    ``ʕa Falaṣṭīn`` "to=Palestine" -- and `CompositionalAnalyzer.analyze`
    finds them for the joined string, but the sentence path analysed the
    words one by one and rebuilt the phrase from the parts (PN run,
    proclitic fold), so the confirmed reading never reached a sentence.
    Runs first, before any sentence rule: a confirmed phrase is settled.
    Longest match wins; nothing crosses punctuation.
    """
    out: list[dict[str, Any]] = []
    index = 0
    while index < len(analyses):
        merged = None
        width_used = 1
        for width in range(min(max_width, len(analyses) - index), 1, -1):
            window = analyses[index:index + width]
            if not all(_span_can_continue(window[k], window[k + 1])
                       for k in range(width - 1)):
                continue
            # Written form of the window: a sandhi piece contributes itself,
            # a whole word its full spelling; pieces of one written word are
            # rejoined with the undertie, separate words with their gap.
            def same_word(left, right):
                return (left.get("from_sandhi") and right.get("from_sandhi")
                        and left.get("source_token") == right.get("source_token")
                        and right.get("source_gap_before") == "")
            partial = any(w.get("from_sandhi") for w in window)
            token = nfc(window[0].get("token") or "")
            source = nfc(window[0].get("source_token") or "")
            for previous, item in zip(window, window[1:]):
                joint = UNDERTIE if same_word(previous, item) else " "
                token += joint + nfc(item.get("token") or "")
                if item.get("source_token") != previous.get("source_token"):
                    gap = item.get("source_gap_before")
                    source += (gap if isinstance(gap, str) and gap.isspace() else " ") \
                        + nfc(item.get("source_token") or "")
            # closing the spaces up is only safe after a bare proclitic
            # (`b ʕÁrabeṯ` recorded as `bʕÁrabeṯ`); `u xōl` is not `uxōl`
            first = nfc(window[0].get("token") or "").lower()
            spelling = has_reading(token, first in proclitics)
            if not spelling:
                continue
            record = analyse(spelling)
            if not _confirmed(record) or not record.get("segmentation"):
                continue
            merged = {**record,
                      "token": token,
                      # a window cut out of a sandhi group shows its own
                      # pieces in line 1 and keeps the whole group as
                      # provenance, exactly like a single sandhi piece
                      "source_token": source,
                      "from_sandhi": partial,
                      "source_gap_before": window[0].get("source_gap_before"),
                      "forced_capital": window[0].get("forced_capital"),
                      "arabic_open": window[0].get("arabic_open"),
                      "piece_start": window[0].get("piece_start"),
                      "piece_end": window[-1].get("piece_end"),
                      "confirmed_span": True}
            width_used = width
            break
        out.append(merged or analyses[index])
        index += width_used
    return out


_VOWELS = set("aeiouāēīōūə")
_CONSTRUCT_L = re.compile(r"=(?:l|lə)$")
_CONSTRUCT_L_GLOSS = re.compile(r"=(?:HD|DOM|DO)$")


def restore_assimilated_l(analyses: list[dict[str, Any]], analyse,
                          is_verbal=None) -> None:
    """Find the =l that the next word swallowed, in place.

    The construct/DOM clitic =l assimilates to the first consonant of the
    following word or merges with its l: awwalčiṭ ṭaləʕṯe, mawhabṯəš šeʕra,
    blōtək kattišō, barrōyṯ‿ṯemme, xwōṯ ləḥcayṯinnaḥ. In Muše Sanžar 13 of 15
    such =l are written this way. Line 2 keeps the author's underlying =l
    (awwal-č=l), line 1 what was said. Tried only for a word the token path
    left weak, and kept only when the restored form reads as =l HD/DOM.
    """
    for index, record in enumerate(analyses[:-1]):
        nxt = analyses[index + 1]
        if record.get("confirmed_span") or record.get("confidence") == "HIGH":
            continue
        current = record.get("segmentation") or ""
        # only a word the token path could not analyse, or read as one
        # undivided piece; a segmented MEDIUM reading (sabek=lun,
        # b=baʕḏ̣=in) is left alone
        if record.get("confidence") != "LOW" and any(m in current for m in "-="):
            continue
        if _CONSTRUCT_L.search(current):
            continue
        same_word = (record.get("from_sandhi") and nxt.get("from_sandhi")
                     and record.get("source_token") == nxt.get("source_token")
                     and nxt.get("source_gap_before") == "")
        if not same_word and not _name_run_can_continue(record, nxt):
            continue
        word = nfc(record.get("token") or "")
        following = nfc(nxt.get("token") or "").lower().lstrip("ə")
        if len(word) < 4 or not following or not word[-1:].isalpha():
            continue
        last = word[-1].lower()
        if last in _VOWELS or last == "l":
            continue
        if following[0] == last:
            # assimilated: stem (+ epenthetic ə/i) + the next word's consonant
            stem = word[:-1]
            if stem[-1:] in ("ə", "i"):
                stem = stem[:-1]
            probe, why = stem + "l", f"=l assimilated to the following {last}"
        elif following.startswith("l"):
            probe, why = word + "l", "=l merged with the following l"
        else:
            continue
        found = analyse(probe)
        segmentation, gloss = found.get("segmentation") or "", found.get("gloss") or ""
        if (found.get("confidence") not in {"HIGH", "MEDIUM"}
                or not _CONSTRUCT_L.search(segmentation)
                or not _CONSTRUCT_L_GLOSS.search(gloss)):
            continue
        host_gloss = _CONSTRUCT_L_GLOSS.sub("", gloss)
        if gloss.endswith("=HD") and (
                (is_verbal and is_verbal(found.get("lemma_id")))
                or re.search(r"\.(SC|NC|PC|PRF|IPV|NC1|NC2)\b", host_gloss)):
            continue          # HD heads a noun phrase: čġayyar=HD is no reading
        placeholder = host_gloss.split("-")[0].lower() in {
            (found.get("lemma") or "").split(" ")[0].lower(),
            re.split(r"[-=]", segmentation)[0].lower()}
        if placeholder and following[0] != last:
            continue          # a host with no meaning + a merged l is too little
        previous = {"segmentation": record.get("segmentation"),
                    "gloss": record.get("gloss"), "tier": record.get("tier"),
                    "lemma": record.get("lemma")}
        record.update({
            "segmentation": segmentation, "gloss": gloss,
            "lemma": found.get("lemma"), "lemma_id": found.get("lemma_id"),
            "lexical_gloss": found.get("lexical_gloss"),
            "tier": found.get("tier"), "confidence": "MEDIUM",
            "alternatives": ([previous] if previous["segmentation"] else [])
            + list(record.get("alternatives") or []),
            "context_rule": why,
            "uncertainty": list(record.get("uncertainty") or []) + [why],
        })


_VERB_NEXT = re.compile(r"(?:^|[-=.])(?:PC|SBJV)(?:$|[-=.])|^[123][MF]?-")


def demonstrative_or_want(analyses: list[dict[str, Any]]) -> None:
    """banna: b=hanna 'in this' before a noun, bann=a 'I want her' before a
    verb. The token path ranks the demonstrative first (4 of 4 in Muše
    Sanžar); only a following subjunctive/PC verb flips it, in place."""
    for index, record in enumerate(analyses[:-1]):
        if not re.match(r"^[^-=]+=h", record.get("segmentation") or ""):
            continue
        want = [a for a in record.get("alternatives") or []
                if str(a.get("lemma") or "").startswith("batt")]
        if not want or not _VERB_NEXT.search(analyses[index + 1].get("gloss") or ""):
            continue
        chosen = want[0]
        previous = {"segmentation": record.get("segmentation"),
                    "gloss": record.get("gloss"), "tier": record.get("tier"),
                    "lemma": record.get("lemma")}
        record.update({"segmentation": chosen["segmentation"], "gloss": chosen["gloss"],
                       "tier": chosen.get("tier"), "lemma": chosen.get("lemma"),
                       "lemma_id": None,
                       "alternatives": [previous] + [a for a in record["alternatives"]
                                                     if a is not chosen],
                       "context_rule": "pseudo-verb: a subjunctive verb follows"})


def _arabic_insert_record(piece: str) -> dict[str, Any]:
    """A word inside an ᴬ…ᴬ Arabic insert: shown, not analysed.

    The author glosses an insert by hand, word or phrase (``Amənsʔalak
    we.will.ask.you``); the analyzer has no Arabic to read it with. Analysed
    as Maalula it came out as noise — the marker A made the opening word a
    capital, so 19 of the 58 insert columns of Muše Sanžar got a ``PN``
    the author had to delete, and a word like ``tisʕa`` could match an
    unrelated Maalula form.
    """
    return {
        "token": piece, "normalized": piece,
        "segmentation_hint": None, "segmentation_hint_satisfied": None,
        "segmentation": None, "gloss": None, "lemma": None, "lemma_id": None,
        "lexical_gloss": None, "confidence": "LOW", "tier": None,
        "alternatives": [], "evidence": [{"kind": "arabic_insert"}],
        "trace": ["inside an ᴬ…ᴬ Arabic insert: not analysed"],
        "uncertainty": ["Arabic insert (ᴬ…ᴬ): gloss by hand"],
        "needs_review": True, "casefold_reason": None,
        "analysis_surface": piece, "arabic_insert": True,
    }


def _capitalized_name_piece(record: dict[str, Any]) -> bool:
    # Eligible member of an automatic multi-word proper-name run.
    if record.get("from_sandhi") or record.get("arabic_open") \
            or record.get("arabic_span"):
        return False
    if record.get("confirmed_span"):
        return False
    raw = nfc(record.get("source_token") or "")
    core = _strip_punctuation(raw)
    if not core or core == "E":
        return False
    # Short capital labels followed by ':'/'.' are speaker tags, not names.
    if raw.rstrip()[-1:] in {":", "."} and len(core) <= 2 \
            and all(ch.isupper() for ch in core if ch.isalpha()):
        return False
    first = core[:1]
    return bool(first and first.isalpha() and first.isupper())


def _name_run_can_continue(left: dict[str, Any], right: dict[str, Any]) -> bool:
    # Only plain whitespace may join the written words of an automatic PN.
    gap = right.get("source_gap_before")
    if gap is not None and not gap.isspace():
        return False
    raw_left = nfc(left.get("source_token") or "").rstrip()
    # Do not jump across comma/period/quotes/etc.; punctuation ends the run.
    if raw_left and unicodedata.category(raw_left[-1]).startswith("P"):
        return False
    return True


def _cannot_start_a_name(record: dict[str, Any],
                         proclitics: frozenset[str]) -> bool:
    """A capital that says nothing about a name.

    * A bare proclitic (``B`` in ``B Lubnān`` at the start of a line) is a
      preposition, not the first word of a name: it folds onto the name
      later (`attach_proclitics`), giving ``b=Lubnān``.
    * A word capitalised only because of its position (line start, after
      . ? ! : or a quote) that the analyzer reads as an ordinary word
      (``Akam Abu Šalhūb``, ``Willa Siryōn``) keeps that reading; the name
      run starts at the next word.
    """
    core = _strip_punctuation(nfc(record.get("source_token") or "")).lower()
    if core in proclitics:
        return True
    if record.get("forced_capital") and record.get("gloss") \
            and record.get("gloss") != "PN" \
            and record.get("confidence") in {"HIGH", "MEDIUM"}:
        return True
    return False


def group_capitalized_proper_names(
        analyses: list[dict[str, Any]],
        proclitics: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    # Merge runs of 2+ capitalized words into one MEDIUM-confidence PN.
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(analyses):
        if (not _capitalized_name_piece(analyses[i])
                or _cannot_start_a_name(analyses[i], proclitics)):
            out.append(analyses[i])
            i += 1
            continue

        run = [analyses[i]]
        j = i + 1
        while (j < len(analyses)
               and _capitalized_name_piece(analyses[j])
               and nfc(_strip_punctuation(analyses[j].get("source_token") or "")).lower()
               not in proclitics
               and _name_run_can_continue(run[-1], analyses[j])):
            run.append(analyses[j])
            j += 1

        if len(run) < 2:
            out.append(analyses[i])
            i += 1
            continue

        merged = copy.deepcopy(run[0])
        source = nfc(run[0].get("source_token") or "")
        forms = [_strip_punctuation(source)]
        for item in run[1:]:
            gap = item.get("source_gap_before")
            joint = gap if isinstance(gap, str) and gap.isspace() else " "
            raw = nfc(item.get("source_token") or "")
            source += joint + raw
            forms.append(_strip_punctuation(raw))

        form = " ".join(part for part in forms if part)
        merged.update({
            "token": form,
            "source_token": source,
            "normalized": form,
            "segmentation": form,
            "gloss": "PN",
            "lemma": form,
            "lemma_id": None,
            "lexical_gloss": None,
            "confidence": "MEDIUM",
            "tier": "C",
            "alternatives": [],
            "evidence": [{"kind": "capitalized_name_run",
                          "forms": forms}],
            "trace": ["2+ adjacent capitalized words -> PN"],
            "uncertainty": ["proper noun inferred from capitalization"],
            "needs_review": True,
            "from_sandhi": False,
        })
        merged["source_gap_before"] = run[0].get("source_gap_before")
        if run[-1].get("piece_end") is not None:
            merged["piece_end"] = run[-1]["piece_end"]
        out.append(merged)
        i = j
    return out


# Spellings no lexicon entry has, read as the spelling it does have. Only
# for lookup: the line the author typed is never changed, and each reading
# says what it was read as.
#
# Cyrillic letters inside a Latin word (Muše Sanžar line 67: aсṯar with a
# Cyrillic с; Žuzēf.Сalaš in a gloss) are typing slips, always.
_CYRILLIC_LOOKALIKES = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
    "і": "i", "А": "A", "Е": "E", "О": "O", "Р": "P", "С": "C", "Х": "X"})
_LATIN = re.compile(r"[A-Za-z\u00C0-\u024F\u1E00-\u1EFF]")
_CYRILLIC = re.compile(r"[\u0400-\u04FF]")
# A stress accent (ḥaláy, minnáy, čʕarfínn) is not in any lexicon entry; the
# author writes it in line 1 and drops it in a segmented line 2
# (ḥal=ayy, minnay=y), keeps it on an undivided word (másalan, ʕÁrabeṯ).
_STRESS = ("\u0301", "\u0300")
# Word-initial ʔ before a vowel, also right after a one-letter proclitic: the
# author writes ʔōb, lʔōbux and glosses ōb, l=ōb=ux. Tried only when the
# written form has no reading, so a real ʔ (ʔumm-ay) is kept.
_GLOTTAL_ONSET = re.compile(r"^([bBlLmMwWuU]?)ʔ(?=[aeiouāēīōūəAEIOUĀĒĪŌŪƏ])")


def _latin_lookalikes(piece: str) -> str:
    if _CYRILLIC.search(piece) and _LATIN.search(piece):
        return piece.translate(_CYRILLIC_LOOKALIKES)
    return piece


def _lookup_variant(piece: str) -> tuple[str | None, str | None]:
    decomposed = unicodedata.normalize("NFD", piece)
    # a stressed final -áy is the author's spelling of the 3 pl -ay=y of a
    # preposition (ʕemmáy -> ʕemmay-y, minnáy -> minnay=y, ruḥláy ->
    # ruḥlay=y in Muše Sanžar; Arnold S. 389-390: ʕlayy, minnayy)
    if decomposed.endswith("a\u0301y"):
        return unicodedata.normalize("NFC", decomposed[:-3] + "ayy"), "stress accent"
    if any(mark in decomposed for mark in _STRESS):
        bare = unicodedata.normalize(
            "NFC", "".join(c for c in decomposed if c not in _STRESS))
        return bare, "stress accent"
    if _GLOTTAL_ONSET.match(piece):
        return _GLOTTAL_ONSET.sub(r"\1", piece, count=1), "initial ʔ"
    return None, None


class TextGlosser:
    def __init__(self, root: Path | str | None = None,
                 analyzer: CompositionalAnalyzer | None = None,
                 productive_fallback: bool = True):
        """``analyzer`` lets the evaluation harness inject a leak-free instance
        while still going through the sentence path, contextual reranking and
        all. ``productive_fallback`` fills tokens the compositional analyzer
        leaves unanalysed with a reconstructing v4 productive analysis; it never
        overrides an existing analysis, so it can only add coverage."""
        self.analyzer = analyzer or CompositionalAnalyzer(root)
        self._root = root
        self._want_fallback = productive_fallback
        self._v4_engine = None
        self._token_cache: dict[tuple, dict[str, Any]] = {}
        # the closed-class inventory decides what may attach; nothing else does
        self.proclitic_forms = frozenset(
            nfc(entry["form"]).lower() for entry in self.analyzer.proclitics)
        # `ni- 1`, `či- 2`, `yi- 3M` and the short series; the person prefixes
        # are morphemes, and a lexicon that happens to spell a word the same way
        # has no business glossing one
        self.verbal_prefixes = {nfc(form): gloss
                                for form, gloss, _meta in self.analyzer.prefixes}

    def remember(self, row: dict[str, Any]) -> None:
        """Take a freshly recorded correction into the live analyzer.

        The corrections layer is read once at construction, so a new reading
        only reached the text after a restart — or after rebuilding the whole
        glosser, which takes seconds the editor does not have on every edit.
        Registering the one row is instant and does exactly what a rebuild
        would: the reading is tier A1 from the next token on.
        """
        surface = nfc(row.get("surface_canonical") or "")
        if not surface:
            return
        self.analyzer.remember_morphology(row)
        # Drop the stale readings and nothing else. Clearing the whole cache
        # would be correct too, but this runs every time the author finishes a
        # word, and the cache is what keeps the editor answering quickly. A
        # piece that merely contains the surface goes as well: `b=waybin` is
        # composed from the host, so its cached answer is stale too.
        aliases = {surface_key(self.analyzer.canonical(s)) for s in morphology_surfaces(row)}
        for key in [k for k in self._token_cache
                    if any(alias in surface_key(self.analyzer.canonical(k[0])) for alias in aliases)]:
            del self._token_cache[key]

    def remember_lexical(self, row: dict[str, Any]) -> None:
        """Take a freshly confirmed lexeme meaning into the live analyzer.

        Sibling of `remember` for the lexical-gloss journal: registers the row
        on `self.analyzer.lexical` so the new meaning is tier AUTHOR from the
        next token on, without rebuilding the analyzer.
        """
        lexeme_id = row.get("lexeme_id")
        if not lexeme_id:
            return
        self.analyzer.lexical.remember(row)
        # Unlike a morphology correction, a lexical one applies to every
        # surface form of the lexeme, so cache entries are dropped by lemma_id
        # rather than by a single surface substring.
        for key, record in list(self._token_cache.items()):
            if record.get("lemma_id") == lexeme_id:
                del self._token_cache[key]

    def _v4(self):
        if self._v4_engine is None and self._want_fallback:
            from maalula_glosser.morphology.analyzer import MorphologyEngine
            try:
                self._v4_engine = MorphologyEngine(self._root or Path.cwd())
            except Exception:
                self._want_fallback = False
        return self._v4_engine

    def _productive_gap_fill(self, piece: str, best: dict,
                             derived_pc_only: bool = False) -> dict | None:
        """A reconstructing v4 analysis; normally only a gap-fill.

        ``derived_pc_only`` is the narrow exception which lets a licensed strong
        II/IV PC beat weak B1/B2/C/D legacy output, never A-tier evidence.
        """
        engine = self._v4()
        if engine is None:
            return None
        # the allophonic subscripts are not part of any stem, and a
        # reconstruction check below compares a segmentation against this form
        piece = fold_allophones(nfc(piece))
        cands = [c for c in engine.analyze(piece) if c.reconstruction_ok]
        if derived_pc_only:
            cands = [c for c in cands
                     if c.features.get("tam") == "SBJV"
                     and c.features.get("derivational_stem")
                     in {"Paʕʕel", "ʔApʕel"}]
        progressive = None
        # Running text fuses the progressive ʕa(m)= with its finite NC host.
        # Let v4 analyse the host, then compose only a reconstructing PRS form;
        # a nominal or another TAM can never license this fallback.
        if not cands:
            for proclitic in sorted(PROGRESSIVE_HOSTS, key=len, reverse=True):
                if not nfc(piece).startswith(proclitic):
                    continue
                host = nfc(piece)[len(proclitic):]
                if len(host) < 3:
                    continue
                host_cands = [
                    c for c in engine.analyze(host)
                    if c.reconstruction_ok and c.features.get("tam") == "PRS"
                ]
                if not host_cands:
                    continue
                host_cands.sort(
                    key=lambda c: 0 if c.evidence == "EXACT_PARADIGM_CELL" else 1)
                progressive = (proclitic, host_cands[0])
                break
        if progressive:
            proclitic, c = progressive
            segmentation = f"{proclitic}={c.segmentation}"
            if re.sub(r"[-=]", "", segmentation) != nfc(piece):
                return None
            return {
                "segmentation": segmentation,
                "gloss": f"PP={render_authoritative_gloss(c.gloss)}",
                "lemma": c.lexeme, "tier": "v4",
                "evidence": ["PRODUCTIVE_PROGRESSIVE", c.evidence],
                "trace": ["v4 progressive fallback", "analyze host", c.evidence],
                "reconstruction_ok": True,
            }
        if not cands:
            return None
        cands.sort(key=lambda c: 0 if c.evidence == "EXACT_PARADIGM_CELL" else 1)
        c = cands[0]
        best_reading = self._v4_reading(c, piece)
        return {**best_reading,
                "tier": "v4", "evidence": [c.evidence],
                "trace": c.trace,
                "reconstruction_ok": True,
                "variants": self._v4_variants(cands[1:], piece, [best_reading])}

    def _v4_reading(self, c, piece: str) -> dict:
        """One v4 candidate as a reading of the word: its gloss with the
        meaning of its lexeme where that lexeme has one."""
        lexeme_id = self._v4_lexeme(c)
        lexical, lexical_status = (
            self.analyzer.lexical.resolve(lexeme_id, piece)
            if lexeme_id else (None, "unknown_lexeme")
        )
        lemma = ((self.analyzer.lexemes.get(lexeme_id) or {}).get("lemma_canonical")
                 if lexeme_id else None) or c.lexeme
        gloss = render_authoritative_gloss(c.gloss)
        # Productive II/IV morphology is generated from a class-specific profile.
        # If that exact lexeme has an English sense, display it rather than the
        # citation form; never borrow a same-root sense from another stem class.
        shown = lexical
        if not shown and str(c.lexeme_id or "").startswith("dict_"):
            # a citation from Arnold's dictionary, in the project's letters
            # (iḳṭaʕ -> ikṭaʕ, as the same verb is cited everywhere else)
            from maalula_glosser.morphology.lexeme_profiles import _ARNOLD_TO_PROJECT
            shown = nfc(c.lexeme or "").translate(_ARNOLD_TO_PROJECT)
        if shown and c.lexeme and c.lexeme in gloss:
            gloss = gloss.replace(c.lexeme, shown, 1)
        return {"segmentation": c.segmentation, "gloss": gloss,
                "lemma": lemma, "lemma_id": lexeme_id,
                "lexical_gloss": lexical,
                "lexical_gloss_status": lexical_status}

    def _v4_variants(self, cands, piece: str, taken: list[dict]) -> list[dict]:
        """The other readings v4 has for the word, to stay on offer: the same
        form cut another way (the DOM after -ell: ell=l or ell INTF.DOM,
        object_transducer.DOM_AFTER_LINKER), another tense or person of the
        same verb (nakš-il=lun NC-MP=3MP beside SC-1S=3MP.IO), another verb.
        """
        seen = {(reading["segmentation"], reading["gloss"]) for reading in taken}
        variants: list[dict] = []
        for other in cands:
            reading = self._v4_reading(other, piece)
            key = (reading["segmentation"], reading["gloss"])
            if key in seen:
                continue
            seen.add(key)
            variants.append({"segmentation": reading["segmentation"],
                             "gloss": reading["gloss"], "tier": "v4",
                             "lemma": reading["lemma"]})
            if len(variants) >= V4_VARIANTS:
                break
        return variants

    def _object_readings_on_offer(self, piece: str) -> list[dict]:
        """A verb with an object suffix stays on offer beside a reading found
        elsewhere: fark-a difference-FREE / fark=a "he separated her",
        č-amr-il=lun "you tell them" / čamr-il=lun (ičmar)."""
        engine = self._v4()
        if engine is None:
            return []
        piece = fold_allophones(nfc(piece))
        offered = self._v4_variants(engine.object_readings(piece), piece, [])
        return [{**reading, "offered": True} for reading in offered]

    def _v4_lexeme(self, c) -> str | None:
        """The project lexeme a v4 reading belongs to.

        A profile built from Arnold's dictionary carries the id of its entry
        (dict_...:I), which is no project lexeme, and one built from the
        training texts carries none, only the gloss it was met with (`buy`
        for iškal, whose lexeme is glossed `take`). Both know their verb by
        its citation form, and the verb lexeme with that citation -- among
        homographs the one of the same root -- gives the lemma and the
        meaning. A dictionary lemma is in Arnold's transcription and is
        respelled first (ixleḳ -> ixlek `be.born`).
        """
        if c.lexeme_id and c.lexeme_id in self.analyzer.lexemes:
            return c.lexeme_id
        from maalula_glosser.morphology.lexeme_profiles import (
            _ARNOLD_TO_PROJECT as ARNOLD_TO_PROJECT)
        engine, profile = self._v4(), None
        if engine is not None:
            profile = (engine.profiles.by_lexeme_id.get(c.lexeme_id) if c.lexeme_id
                       else engine.profiles.by_lemma.get(nfc(c.lexeme or "")))
        if profile is None or profile.pos != "VERB":
            return c.lexeme_id or self._verb_lexeme_for(c.lexeme)
        dictionary = str(profile.lexeme_id or "").startswith("dict_")
        lemma = nfc(profile.lemma or "")
        root = "".join(profile.root or ())
        entry = str(profile.lexeme_id or "").split(":")[0]
        for citation in (profile.slot("PST_CITATION"),
                         lemma.translate(ARNOLD_TO_PROJECT) if dictionary else lemma):
            found = self._verb_lexemes(citation)
            if len(found) > 1:
                # homographs (šaʕʕel "ask" and šaʕʕel "kindle"): the one of
                # the profile's root, else the one its dictionary entry was
                # matched to; with neither the citation stays, not a guess
                same_root = [lexeme_id for lexeme_id in found if root and nfc(str(
                    (self.analyzer.lexemes.get(lexeme_id) or {})
                    .get("root_canonical") or "")) == root]
                linked = getattr(self.analyzer, "lexeme_by_source_entry", {}).get(entry)
                found = (same_root if len(same_root) == 1
                         else [linked] if linked in found else [])
            if found:
                return found[0]
        return c.lexeme_id

    def _verb_lexemes(self, citation: str | None) -> list[str]:
        """The verb lexemes whose citation begins with this form."""
        if not citation:
            return []
        index = getattr(self, "_verb_heads", None)
        if index is None:
            index = {}
            verbal = self.analyzer.verbal_lexeme_ids
            for lexeme_id, lexeme in self.analyzer.lexemes.items():
                head = nfc(str(lexeme.get("lemma_canonical") or "")).split(" ")[0]
                if head and (lexeme_id in verbal or lexeme_id.startswith(("lexnew_", "lexsvc_"))):
                    index.setdefault(head, []).append(lexeme_id)
            self._verb_heads = index
        return list(index.get(nfc(citation)) or [])

    def _verb_lexeme_for(self, citation: str | None) -> str | None:
        """A v4 profile without a lexicon link (an Arnold-only verb such as
        šaġġel, raḥḥem) names its lexeme by citation form: find the verb
        lexeme whose citation starts with it, so the gloss can show the
        meaning (`employ`, `have.mercy`) and the word gets its lemma."""
        found = self._verb_lexemes(citation)
        return found[0] if found else None

    def _compose_typed_boundaries(self, piece: str) -> dict[str, Any] | None:
        """Analyse each piece the author cut off, when the whole form is unknown.

        A `-` or `=` in the input is a statement about where the morphemes are.
        Until now it was used only to *prefer* an analysis of the whole word, so
        when the whole word was unknown the statement was thrown away: `w-la`
        came back `??-??` although `w` is `and` and `la` is `NEG`, both of them
        in the closed-class inventory.

        A piece that resolves contributes its label; one that does not leaves an
        empty cell, so a half-known word says what it knows instead of nothing.
        The result is offered only if at least one piece was recognised.
        """
        text = fold_allophones(nfc(piece))
        parts = [p for p in re.split(r"([-=])", text) if p]
        if len(parts) < 3:
            return None
        # [form, sep, form, sep, form ...]
        forms = parts[0::2]
        separators = parts[1::2]
        if not all(forms) or len(separators) != len(forms) - 1:
            return None

        segments, labels, resolved = [], [], 0
        for position, form in enumerate(forms):
            # A first piece with material after it is a prefix, not a word.
            # `yi-ščwit` came back `yiʔ-` because the person prefix was looked
            # up in the general lexicon, which spells a word that way; the
            # prefix inventory is the only place it can honestly come from.
            if not position and separators and separators[0] == "-" \
                    and form in self.verbal_prefixes:
                segments.append(form)
                labels.append(third_person(self.verbal_prefixes[form], text))
                resolved += 1
                continue
            analysis = self.gloss_token(form, compose=False)
            segmentation = analysis.get("segmentation") or ""
            gloss = analysis.get("gloss") or ""
            # Everything after the first piece is an affix, a clitic or a host,
            # and an affix read as a free-standing word picks up whatever the
            # lexicon happens to spell that way: `mḥammṣ-in=le` came back
            # `-in=affair` where the pieces are the plural suffix and the 3ms
            # object. Only direct evidence counts there — the closed class, the
            # confirmed corpus, and the name heuristic, since a capitalised
            # piece after a clitic boundary (`b=Amērca`) really is a name. The
            # first piece is the stem, where a lexicon reading is the point.
            if position and analysis.get("tier") not in ("A0", "A1", "A2", "A3", "C"):
                segmentation = gloss = ""
            # a piece whose own two lines disagree cannot be spliced into a row
            # that has to stay aligned; it contributes its bare form instead
            if segmentation and gloss and \
                    re.findall(r"[-=]", segmentation) == re.findall(r"[-=]", gloss):
                segments.append(segmentation.rstrip("-="))
                labels.append(gloss.rstrip("-="))
                resolved += 1
            else:
                segments.append(form)
                labels.append("")
        options = typed_bound_options(
            list(zip([""] + separators, forms)), self, labels)
        for index, choices in enumerate(options):
            if choices and (not labels[index]
                            or labels[index].upper() == "NEG" and separators[index - 1:index] == ["="]):
                if not labels[index]:
                    resolved += 1
                labels[index] = choices[0]
        if not resolved:
            return None
        segmentation = segments[0]
        gloss = labels[0]
        for separator, segment, label in zip(separators, segments[1:], labels[1:]):
            segmentation += separator + segment
            gloss += separator + label
        return {
            "segmentation": segmentation,
            "gloss": gloss,
            "tier": "B1",
            "evidence": [{"kind": "typed_boundaries", "pieces": forms}],
            "trace": ["whole form unknown", "composed from the boundaries typed"],
            "reconstruction_ok": True,
        }

    def gloss_token(self, piece: str, sentence_initial: bool = False,
                    context: dict | None = None,
                    compose: bool = True) -> dict[str, Any]:
        # A word the analyzer knows nothing about costs seconds, because the
        # productive generator has to try everything before giving up, and the
        # editor asks about the same word again on every boundary the author
        # moves. Remembered for the life of the process — but only when no
        # scoped correction is in play, since that answer depends on where the
        # token sits. A copy goes out: callers write on what they are handed.
        key = (piece, sentence_initial, compose) if context is None else None
        if key is not None and key in self._token_cache:
            return copy.deepcopy(self._token_cache[key])
        latin = _latin_lookalikes(piece)
        if latin != piece:
            record = self.gloss_token(latin, sentence_initial, context, compose)
            record["token"] = piece
            record["spelling_variant"] = {"read_as": latin, "reason": "Cyrillic letter"}
            record["uncertainty"] = [*record.get("uncertainty", []),
                                     f"Cyrillic letter in a Latin word: read as {latin}"]
            if key is not None:
                self._token_cache[key] = copy.deepcopy(record)
            return record

        # At a punctuation-conditioned capital, analyse the ordinary lower-case
        # word first.  LOW is still unresolved, so in that case retain the old
        # capitalised path (including PN fallback) rather than allowing a weak
        # dictionary guess to steal a genuine name.
        working_piece = piece
        casefold_reason = None
        result = None
        if sentence_initial and piece[:1].isupper():
            lowered = piece[:1].lower() + piece[1:]
            lowered_result = self.analyzer.analyze(
                lowered, sentence_initial=False, context=context, _normal_only=True)
            if (lowered_result.get("best")
                    and lowered_result["best"].get("gloss") != "PN"
                    and lowered_result.get("confidence") in {"HIGH", "MEDIUM"}):
                result = lowered_result
                working_piece = lowered
                casefold_reason = "punctuation-conditioned capitalization"
                original_result = self.analyzer.analyze(
                    piece, sentence_initial=sentence_initial,
                    context=context, _normal_only=True)
                original_best = original_result.get("best") or {}
                lower_best = lowered_result["best"]
                # Orthographic normalization cannot override a different
                # author-confirmed or normative reading of the written form.
                if (original_best.get("tier") in {"A1", "A0"}
                        and ((original_best.get("segmentation") or "").lower(),
                             original_best.get("gloss"))
                        != ((lower_best.get("segmentation") or "").lower(),
                            lower_best.get("gloss"))):
                    result = original_result
                    working_piece = piece
                    casefold_reason = None
        if result is None:
            result = self.analyzer.analyze(piece, sentence_initial=sentence_initial,
                                           context=context, _normal_only=True)
        best = result.get("best") or {}
        # productive gap-fill: fire when the compositional analyzer segmented
        # nothing, or left only a weak undecomposed whole-form reading (a
        # recognised lexeme with no internal morphology, e.g. mižčamʕin -> the
        # bare lemma ižčmaʕ). A decomposing productive analysis is strictly
        # better there; a good segmented analysis is never touched.
        v4_filled = False
        seg = best.get("segmentation")
        weak_tier = best.get("tier") in ("B1", "B2", "C", "D")
        undecomposed_weak = (seg and "-" not in seg and "=" not in seg
                             and weak_tier)
        # A reading with an object clitic directly after the DOM is no better
        # than an undivided one: v4 is asked for the word, and that reading
        # stays on offer last.
        after_dom = any(isinstance(note, dict) and note.get("type") == "object_after_dom"
                        for note in best.get("uncertainties") or [])
        fill = None
        if not seg or undecomposed_weak or after_dom:
            fill = self._productive_gap_fill(working_piece, best)
        elif weak_tier:
            # Productive strong II/IV PC may replace only weak legacy evidence.
            fill = self._productive_gap_fill(working_piece, best, derived_pc_only=True)
        # when replacing an existing reading, require a real decomposition
        if fill and (not seg or "-" in fill["segmentation"]
                     or "=" in fill["segmentation"]):
            variants = fill.pop("variants", None) or []
            kept = ([{"segmentation": seg, "gloss": best.get("gloss"),
                      "tier": best.get("tier"), "lemma": best.get("lemma")}]
                    if after_dom else [])
            best = {**best, **fill}
            result["confidence"] = "MEDIUM"
            if variants or kept:
                result["alternatives"] = (
                    variants + list(result.get("alternatives") or []) + kept)
            v4_filled = True
        # A capitalized token gets the same productive v4 opportunity as its
        # lowercase spelling before a name can be proposed.
        if (piece[:1].isupper() and not casefold_reason and not v4_filled
                and result.get("confidence") not in {"HIGH", "MEDIUM"}):
            lowered = piece[:1].lower() + piece[1:]
            fill = self._productive_gap_fill(lowered, best)
            if fill:
                fill.pop("variants", None)
                best = {**best, **fill}
                result["confidence"] = "MEDIUM"
                result["alternatives"] = []
                v4_filled = True
        # Read the boundaries the author typed when nothing else produced a
        # segmentation — or when all it produced was an undivided guess.
        # `y-affkull` was answered `yaffkull go`, a whole-form reading with no
        # person and no TAM in it, and the cut the author had just typed was
        # dropped on the floor. A human decision outranks generation.
        settled = best.get("segmentation") or ""
        undivided = settled and not any(mark in settled for mark in "-=")
        composed = False
        if compose and (not settled or undivided) and any(m in working_piece for m in "-="):
            fallback = self._compose_typed_boundaries(working_piece)
            if fallback:
                best = {**best, **fallback}
                result["confidence"] = "MEDIUM"
                composed = v4_filled = True
        # Not at a punctuation-conditioned capital: there every word is
        # capitalised, so the capital says nothing about a name. Such a word
        # stays unanalysed rather than become PN (the analyzer withholds the
        # same guess in `analyze`; this second fallback used to bring it back).
        if (piece[:1].isupper() and not v4_filled and not sentence_initial
                and result.get("confidence") not in {"HIGH", "MEDIUM"}):
            names = self.analyzer._as_proper_noun(
                result.get("normalized") or piece, nfc(piece).lstrip("-="))
            if names:
                best = names[0].to_dict()
                result["confidence"] = "MEDIUM"
                result["alternatives"] = []
        if best.get("gloss") and not v4_filled:
            best["gloss"] = render_authoritative_gloss(best["gloss"])
        if best.get("segmentation") and not v4_filled:
            offered = self._object_readings_on_offer(working_piece)
            # Arnold cites the 3 sg m preterite before a noun with its -il
            # (laḳṭil "prät. 3 sg. m."), and that citation came back as a bare
            # preterite, lakṭil seize.SC. The word is the same verb with the
            # DOM: lakṭ=l seize.SC=DOM (Arnold § 5.12; so the texts cut it).
            marked = next((
                reading for reading in offered
                if best.get("tier") == "A2" and "=" not in best["segmentation"]
                and "-" not in best["segmentation"]
                and str(best.get("gloss") or "").endswith(".SC")
                and reading["gloss"] == best["gloss"] + "=DOM"
                and reading["segmentation"].endswith("=l")), None)
            alternatives = list(result.get("alternatives") or [])
            if marked:
                offered.remove(marked)
                alternatives.insert(0, {
                    "segmentation": best["segmentation"], "gloss": best["gloss"],
                    "tier": best.get("tier"), "lemma": best.get("lemma")})
                best = {**best, "segmentation": marked["segmentation"],
                        "gloss": marked["gloss"], "tier": "v4",
                        "evidence": ["PRODUCTIVE_OBJECT_DOM", *best.get("evidence", [])],
                        "reconstruction_ok": True}
                result["confidence"] = "MEDIUM"
            result["alternatives"] = alternatives + offered
        if best.get("gloss"):
            best["gloss"] = nc_plural_participle(
                best.get("segmentation"), sc_plural_person(best["gloss"]))
            best["segmentation"] = iv_person_prefix(
                best.get("segmentation"), best["gloss"])
        for alternative in result.get("alternatives", []):
            if alternative.get("gloss"):
                alternative["gloss"] = nc_plural_participle(
                    alternative.get("segmentation"),
                    sc_plural_person(render_authoritative_gloss(alternative["gloss"])))
                alternative["segmentation"] = iv_person_prefix(
                    alternative.get("segmentation"), alternative["gloss"])
        # an offered reading that only repeats one already there is dropped
        # (compared after the labels above are in their final shape)
        if any(a.get("offered") for a in result.get("alternatives", [])):
            shown = {(best.get("segmentation"), best.get("gloss"))}
            kept = []
            for alternative in result["alternatives"]:
                key = (alternative.get("segmentation"), alternative.get("gloss"))
                if alternative.pop("offered", False) and key in shown:
                    continue
                shown.add(key)
                kept.append(alternative)
            result["alternatives"] = kept
        uncertainty = []
        if composed:
            uncertainty.append("whole form unknown: read from your boundaries")
        elif result["confidence"] != "HIGH":
            uncertainty.append(
                "no analysis found" if not best else
                ("several competing analyses" if result["n_candidates"] > 1
                 else "weak evidence tier"))
        for note in best.get("uncertainties") or []:
            if isinstance(note, dict) and note.get("type") == "verbal_label_on_a_noun":
                uncertainty.append(note["reason"])
        if best.get("lexical_gloss_status") in ("translation_pending", "no_sense",
                                                "unknown_lexeme"):
            uncertainty.append(f"lexical gloss {best['lexical_gloss_status']}")
        if best and not best.get("reconstruction_ok"):
            uncertainty.append("surface not fully reconstructed")
        if result.get("segmentation_hint") and not result.get("segmentation_hint_satisfied"):
            uncertainty.append("your segmentation could not be reproduced")
        record = {
            "token": piece,
            "normalized": result["normalized"],
            "segmentation_hint": result.get("segmentation_hint"),
            "segmentation_hint_satisfied": result.get("segmentation_hint_satisfied"),
            "segmentation": best.get("segmentation"),
            "gloss": best.get("gloss"),
            "lemma": best.get("lemma"),
            "lemma_id": best.get("lemma_id"),
            "lexical_gloss": best.get("lexical_gloss"),
            "confidence": result["confidence"],
            "tier": best.get("tier"),
            "alternatives": [
                {"segmentation": a["segmentation"], "gloss": a["gloss"],
                 "tier": a["tier"], "lemma": a.get("lemma")}
                for a in result.get("alternatives", [])
            ],
            "evidence": best.get("evidence", []),
            "trace": best.get("trace") or [
                "legacy compositional analyzer",
                *[str(x) for x in best.get("evidence", [])],
            ],
            "uncertainty": uncertainty,
            "needs_review": result["needs_review"],
            "casefold_reason": casefold_reason,
            "analysis_surface": working_piece,
        }
        if (not record["segmentation"] or record["confidence"] == "LOW"
                or (record["gloss"] == "PN" and record["tier"] == "C")):
            variant, reason = _lookup_variant(piece)
            if variant:
                other = self.gloss_token(variant, sentence_initial, context, compose)
                # Only a reading the corpus, the closed class or a printed
                # Arnold paradigm vouches for: a
                # productive guess on the respelled word is often wrong in a
                # new way (ʕemmáy is ʕemmay=y with=3P, read bare it became
                # ʕemm-ay with-MS.INDF), and an empty cell is better than that.
                vouched = other.get("tier") in ("A0", "A1", "A2", "A3") or any(
                    isinstance(item, dict) and item.get("kind") == "ARNOLD_PARADIGM_CELL"
                    for item in other.get("evidence") or [])
                if other.get("segmentation") and other["confidence"] in ("HIGH", "MEDIUM") \
                        and vouched:
                    segmentation = other["segmentation"]
                    if reason == "stress accent" and "-" not in segmentation \
                            and "=" not in segmentation:
                        segmentation = piece      # másalan: undivided keeps it
                    record = {**other, "token": piece, "segmentation": segmentation,
                              "confidence": "MEDIUM" if other["confidence"] == "HIGH"
                              else other["confidence"],
                              "spelling_variant": {"read_as": variant, "reason": reason},
                              "uncertainty": [*other.get("uncertainty", []),
                                              f"{reason}: read as {variant}"]}
        if key is not None:
            # bounded: a long session must not grow a second copy of the corpus
            if len(self._token_cache) >= TOKEN_CACHE:
                self._token_cache.clear()
            self._token_cache[key] = copy.deepcopy(record)
        return record

    def gloss_sentence(self, sentence: str, document_id: str | None = None,
                       sentence_id: str | None = None) -> dict[str, Any]:
        from maalula_glosser.editor import arabic_marks
        analyses = []
        arabic_spans = arabic_marks.spans(sentence)
        arabic_openers = {open_ for open_, _close in arabic_spans}
        source_cursor = 0
        piece_counter = 0
        for index, token in enumerate(tokenize(sentence)):
            # tokenize() deliberately drops punctuation-only items such as
            # "...". Remember the literal gap between surviving words so a
            # proclitic cannot jump across that punctuation later.
            source_start = sentence.find(token["original"], source_cursor)
            if source_start >= 0:
                source_gap = sentence[source_cursor:source_start]
                source_cursor = source_start + len(token["original"])
            else:
                # Defensive fallback: old behaviour is safer than inventing a
                # punctuation boundary if an exotic spelling cannot be found.
                source_gap = None
            for piece_index, piece in enumerate(token["pieces"]):
                # A scoped token correction binds to (document, sentence, token
                # index); only a caller that supplies a document_id can match
                # one, so arbitrary user text never triggers a scoped override.
                context = None
                if document_id is not None:
                    context = {"document_id": document_id,
                               "sentence_id": sentence_id if sentence_id is not None
                               else "0",
                               "token_index": index,
                               "original_token": token["original"]}
                forced_capital = ((index == 0 and piece_index == 0)
                                  or _contextual_capital(sentence, token, piece_index))
                inside_arabic = source_start >= 0 and any(
                    open_ < source_start + len(token["original"]) and source_start <= close
                    for open_, close in arabic_spans)
                record = self.gloss_token(
                    piece, sentence_initial=forced_capital, context=context)
                if inside_arabic:
                    # A Maalula word the corpus or the closed class knows may
                    # stand inside an insert too (u, ʕammo); anything weaker —
                    # a capital-letter PN, a productive guess (xams=a for the
                    # Arabic numeral xamsa) — is noise the author would delete.
                    if record.get("tier") not in ("A0", "A1", "A2", "A3"):
                        record = _arabic_insert_record(piece)
                    record["arabic_span"] = True
                record["forced_capital"] = forced_capital
                record["source_token"] = token["original"]
                record["from_sandhi"] = token["sandhi"]
                # Which written pieces (between spaces and underties, counted
                # over the whole line) this record stands for; merges below
                # widen the span. `session.rewrite_line` uses it to take
                # contextual readings for just the pieces that changed.
                record["piece_start"] = record["piece_end"] = piece_counter
                piece_counter += 1
                # Only the first analysis from a written token owns the gap
                # before that token. Sandhi pieces inside one written token
                # remain adjacent to each other.
                record["source_gap_before"] = source_gap if piece_index == 0 else ""
                record["arabic_open"] = bool(
                    piece_index == 0 and source_start in arabic_openers)
                analyses.append(record)
        analyses = apply_confirmed_spans(
            analyses, self.analyzer.has_confirmed_reading, self.gloss_token,
            proclitics=self.proclitic_forms)
        analyses = group_capitalized_proper_names(analyses, self.proclitic_forms)
        disambiguate_progressive(analyses)
        apply_constructions(analyses)
        group_mwe(analyses)
        demonstrative_or_want(analyses)
        restore_assimilated_l(analyses, self.gloss_token,
                              lambda lemma_id: lemma_id in self.analyzer.verbal_lexeme_ids)
        # last, so the passes above still see the two tokens apart
        analyses = attach_proclitics(analyses, self.proclitic_forms,
                                     self.gloss_token)
        return {
            "original": sentence,
            "tokens": analyses,
            "segmentation_line": " ".join(a["segmentation"] or a["normalized"]
                                          for a in analyses),
            "gloss_line": " ".join(a["gloss"] or "???" for a in analyses),
            "confidence_counts": {
                level: sum(1 for a in analyses if a["confidence"] == level)
                for level in ("HIGH", "MEDIUM", "LOW")},
        }

    def gloss_text(self, text: str) -> list[dict[str, Any]]:
        return [self.gloss_sentence(s) for s in split_sentences(text)]


def format_igt(sentence: dict[str, Any]) -> str:
    """Readable four-line interlinear block."""
    marks = {"HIGH": " ", "MEDIUM": "~", "LOW": "?"}
    seg = [a["segmentation"] or a["normalized"] for a in sentence["tokens"]]
    gl = [a["gloss"] or "???" for a in sentence["tokens"]]
    cf = [marks[a["confidence"]] for a in sentence["tokens"]]
    widths = [max(len(s), len(g), 1) for s, g in zip(seg, gl)]
    lines = [
        sentence["original"],
        "  ".join(s.ljust(w) for s, w in zip(seg, widths)),
        "  ".join(g.ljust(w) for g, w in zip(gl, widths)),
        "  ".join(c.ljust(w) for c, w in zip(cf, widths)),
    ]
    counts = sentence["confidence_counts"]
    lines.append(f"[HIGH {counts['HIGH']} / MEDIUM {counts['MEDIUM']} / LOW {counts['LOW']}"
                 "   ~ = review suggested, ? = unresolved]")
    return "\n".join(lines)


def format_table(sentences: list[dict[str, Any]],
                 show_alternatives: bool = False,
                 show_trace: bool = False) -> str:
    """Plain terminal table for readers who do not use IGT conventions yet."""
    headers = ("TOKEN", "SEGMENTATION", "LEMMA", "GLOSS", "LANG",
               "MORPHOLOGY", "CONFIDENCE", "SOURCE")
    rows = []
    alternatives = []
    for sentence in sentences:
        for token in sentence["tokens"]:
            language = token.get("gloss_language") or "und"
            lexical = token.get("lexical_gloss") or "UNKNOWN"
            displayed = f"{lexical} [de]" if language == "de" else lexical
            source = token.get("morph_engine") or token.get("tier") or "UNKNOWN"
            lexical_source = token.get("gloss_source")
            if lexical_source:
                source = f"{source}/{lexical_source}"
            rows.append((
                token.get("token") or "",
                token.get("segmentation") or "UNKNOWN",
                token.get("lemma") or "UNKNOWN",
                displayed,
                language,
                token.get("morphology") or "UNKNOWN",
                token.get("confidence") or "LOW",
                source,
            ))
            if show_alternatives:
                for alt in token.get("alternatives") or []:
                    alternatives.append(
                        f"  ↳ {token.get('token')}: "
                        f"{alt.get('segmentation') or 'UNKNOWN'} | "
                        f"{alt.get('gloss') or 'UNKNOWN'} | "
                        f"lemma={alt.get('lemma') or 'UNKNOWN'} | "
                        f"source={alt.get('tier') or 'UNKNOWN'}")
            if show_trace:
                alternatives.append(
                    f"  ↳ {token.get('token')}: TRACE "
                    + " → ".join(str(x) for x in token.get("trace") or ["no trace"]))
    widths = [max(len(str(value)) for value in (header, *(r[i] for r in rows)))
              for i, header in enumerate(headers)]

    def line(values):
        return "  ".join(str(value).ljust(width) for value, width in zip(values, widths))

    output = [line(headers), line(tuple("-" * width for width in widths))]
    output.extend(line(row) for row in rows)
    if alternatives:
        output.extend(("", "ALTERNATIVES", *alternatives))
    return "\n".join(output)


def build_review_queue(sentences: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """MEDIUM/LOW tokens, ready for the append-only review store."""
    queue = []
    for index, sentence in enumerate(sentences):
        for position, token in enumerate(sentence["tokens"]):
            if token["confidence"] == "HIGH":
                continue
            queue.append({
                "review_id": f"gtq_{index:04d}_{position:03d}",
                "sentence": sentence["original"],
                "token": token["token"],
                "normalized": token["normalized"],
                "proposed_segmentation": token["segmentation"],
                "proposed_gloss": token["gloss"],
                "lemma": token["lemma"],
                "confidence": token["confidence"],
                "segmentation_hint_satisfied": token.get("segmentation_hint_satisfied"),
                "alternatives": token["alternatives"],
                "evidence": token["evidence"],
                "uncertainty": token["uncertainty"],
                "status": "open",
            })
    return queue
