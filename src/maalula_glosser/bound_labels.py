"""Context-sensitive labels for morphemes separated by an author.

Free words and bound morphemes share spellings in Maalula: ``la`` is NEG as a
word but a pronominal exponent after ``=``; ``a`` can be verbal agreement but
is normally FREE at the edge of a noun.  This module is deliberately small and
works only after the author has supplied boundaries.  It does not segment an
unknown word or infer a lexeme.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable


TAM_RE = re.compile(
    r"(?:^|[.=+\-])(?:NC|NC1|NC2|SC|PC|PRF|PRS|PST|SBJV|IPV)(?=$|[.=+\-])")
OBJECT_CLITICS = {
    "le": ("3MS", "3MS.IO"),
    "la": ("3FS", "3FS.IO"),
    "lun": ("3MP", "3MP.IO"),
}


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value or "")


def _is_verbal(parts: list[tuple[str, str]], labels: list[str], glosser) -> bool:
    if any(TAM_RE.search(label or "") for label in labels):
        return True
    analyzer = getattr(glosser, "analyzer", None)
    prefixes = {nfc(form) for form, _gloss, _meta
                in getattr(analyzer, "prefixes", [])}
    return bool(len(parts) > 1 and parts[0][1] in prefixes
                and parts[1][0] == "-")


def typed_bound_options(parts: Iterable[tuple[str, str]], glosser=None,
                        existing: Iterable[str] | None = None, *,
                        preserve_object_roles: bool = False) -> list[list[str]]:
    """Ordered label choices for a complete, manually segmented token.

    The first item is the default suggestion.  Further items are alternatives
    for the editor.  Existing matching analyses win except where treating a
    bound exponent as a free word is structurally impossible.
    """
    pieces = [(sep, nfc(form)) for sep, form in parts]
    labels = list(existing or [])
    labels += [""] * (len(pieces) - len(labels))
    options: list[list[str]] = [[label] if label else [] for label in labels]
    verbal = _is_verbal(pieces, labels, glosser)

    for index, (sep, form) in enumerate(pieces):
        following = pieces[index + 1] if index + 1 < len(pieces) else None

        # These are nominal chains only after the author supplied both cuts.
        if sep == "-" and form in {"ṯ", "č"} and following:
            if following[0] in {"-", "=", "+"}:
                options[index] = ["F"]
        if sep == "-" and form in {"ōṯ", "yōṯ"} and following:
            if following[0] in {"-", "=", "+"}:
                options[index] = ["P", "FP"]
        if sep == "-" and form == "a" and index == len(pieces) - 1:
            options[index] = ["F"] if verbal else ["FREE", "F"]

        # In an NC object form -il/-in precedes the pronominal enclitic.  An
        # exact full analysis (e.g. amr-il=lun, SC-1S=3MP.IO) has already filled
        # this cell and must not be replaced by the productive default.
        if (sep == "-" and form in {"il", "in"} and following
                and following[0] in {"=", "+"} and following[1] in OBJECT_CLITICS
                and not options[index]):
            options[index] = ["MP"]

        follows_verbal_exponent = bool(
            index and pieces[index - 1][0] == "-"
            and pieces[index - 1][1] in {"ō", "il", "in", "al"})
        if sep in {"=", "+"} and form in OBJECT_CLITICS and (verbal or follows_verbal_exponent):
            current = options[index][0] if options[index] else ""
            # A standalone lookup of =la as NEG is invalid in this position.
            if preserve_object_roles and current in OBJECT_CLITICS[form]:
                options[index] = [current]
            elif current.upper() == "NEG" or not current \
                    or current in OBJECT_CLITICS[form]:
                options[index] = list(OBJECT_CLITICS[form])
            else:
                # A confirmed full-token label is authoritative, including IO.
                options[index] = [current]

    # PP is licensed only before an NC/PRS host.  A nominal/PN host and every
    # other verbal TAM must fall back to the ordinary prepositional readings.
    nc_verbal = any(re.search(
        r"(?:^|[.=+\-])(?:NC|PRS)(?=$|[.=+\-])", label or "")
        for label in labels
    )
    if pieces and pieces[0] == ("", "ʕa") and len(pieces) > 1 \
            and pieces[1][0] in {"=", "+"} and not nc_verbal:
        analyzer = getattr(glosser, "analyzer", None)
        prepositional = []
        for entry in getattr(analyzer, "proclitics", []):
            if nfc(entry.get("form") or "") != "ʕa":
                continue
            label = nfc(entry.get("gloss") or "").rstrip("=")
            if label and label != "PP" and label not in prepositional:
                prepositional.append(label)
        if prepositional:
            options[0] = prepositional

    return options
