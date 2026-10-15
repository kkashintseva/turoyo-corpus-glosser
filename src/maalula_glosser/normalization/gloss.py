from __future__ import annotations

import re
from collections.abc import Mapping

# OCR/text-extraction often turns Latin small-caps into Cyrillic lookalikes.
LOOKALIKE_TRANSLATION = str.maketrans({
    "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "К": "K",
    "М": "M", "О": "O", "Р": "P", "Т": "T", "Х": "X",
    "а": "a", "с": "c", "е": "e", "о": "o", "р": "p", "х": "x",
})

# Leipzig-style person/gender/number bundles and the two adjective-status labels
# used in the project. Keep this in one place: normalization, rule parsing and
# the discrepancy engine must agree on what is a grammatical feature.
PERSON_FEATURE_RE = re.compile(
    r"^(?:[123](?:M|F|C)?(?:S|P)?|M|F|C|S|P|MS|FS|MP|FP|DEF|INDF)$",
    re.IGNORECASE,
)
SEPARATOR_RE = re.compile(r"([-+=.])")

# FS_SHABO_LAST — копия.docx, section II, defines SC, PC and NC as the
# canonical IGT labels.  They are conjugation categories, not aliases for tense
# labels.  Internal morphology may still carry PST/PRS/SBJV as ``features.tam``,
# but normalizing an IGT gloss must not erase the source category.  The editor
# currently permits the two perfect subclasses NC1/NC2 to be displayed as PRF.
DEFAULT_GLOSS_ALIASES: dict[str, str] = {
    "NC1": "PRF",
    "NC2": "PRF",
    "NC 1": "PRF",
    "NC 2": "PRF",
    "PRES": "NC",
    "PERF": "PRF",
    "SUBJ": "PC",
    "IMP": "IPV",
    "CSTR": "CNS",
    "SG": "S",
    "PL": "P",
}

_DISPLAY_CONJUGATION = {"PST": "SC", "PRS": "NC", "SBJV": "PC"}
AUTHORITATIVE_CONJUGATION_LABELS = {"SC", "PC", "NC"}


def render_authoritative_gloss(gloss: str) -> str:
    """Render finite verb categories using the authoritative IGT convention.

    Only labels following a lexical gloss dot are converted.  This deliberately
    leaves the independently attested ``pst-be`` analysis of the past copula
    untouched.
    """
    return re.sub(
        r"(?<=[./])(PST|PRS|SBJV)\b",
        lambda match: _DISPLAY_CONJUGATION[match.group(1).upper()],
        gloss,
        flags=re.IGNORECASE,
    )


# `do.nc-mp`, `1-do.pc-intf=3ms`: the verb išw. The label DO (direct object:
# lē=le DO=3MS) stands alone in its cell and never carries a conjugation label,
# so a cell that opens with DO and goes on after a dot is the lexical verb.
_LEXICAL_DO = re.compile(r"(?<![^-=+\s])DO(?=\.)")


def lexical_do(gloss: str) -> str:
    """`DO.NC-MP` -> `do.NC-MP`: the author's `do`, which the label
    normaliser took for DO (the author, 02.10.2026: "do, не DO")."""
    if not gloss or "DO." not in gloss:
        return gloss
    return _LEXICAL_DO.sub("do", gloss)


_SC_PLURAL = re.compile(r"(?<=[./])(SC-)([MF]P)(?=$|[-=+/.\s])", re.IGNORECASE)


def sc_plural_person(gloss: str) -> str:
    """A suffix-conjugation plural carries its person: ``SC-3MP``, never
    ``SC-MP`` (the author, 30.09.2026: aḥəfẓ-un-n=a remember.SC-3mp-INTF=3fs,
    "-un- в SC глоссируется как 3mp, вопреки некоторым глоссам"). The
    prefix conjugation keeps ``-MP``: its person is on the prefix
    (y-lufẓ-un-n=a 3-pronounce.PC-MP-INTF=3fs). Applied to what the parser
    proposes, not to glosses already saved in a text.
    """
    if not gloss:
        return gloss
    return _SC_PLURAL.sub(
        lambda m: m.group(1) + ("3" + m.group(2).lower()
                                if m.group(2).islower() else "3" + m.group(2)),
        gloss)


_NC_PERSON = re.compile(r"^3([MF]P)$", re.IGNORECASE)
_PARTICIPLE_PLURAL = re.compile(r"[iī]n$")


_IV_PERSON_PREFIX = re.compile(r"^([yčn])a-(?=[^-=])")
_PERSON_LABEL = re.compile(r"^[123][MFmf]?$")


def iv_person_prefix(segmentation: str | None, gloss: str | None) -> str | None:
    """IV in the PC: the person prefix is the bare consonant and the class
    ``a-`` stays with the stem, as Shabo cuts it: ``y-aktr-un 3-can.pc-mp``,
    ``y-aḥc-un``, ``č-aḥəc 3f-speak.pc``, ``n-aylf-ell=l`` (FS_SHABO).

    ``glossed_all`` cut the same words ``ya-ktr-un``, ``ya-ḥc-un``,
    ``ya-rəḥm-enn=e``; the author ruled for Shabo (01.10.2026). A PC prefix
    has no vowel ``a`` of its own (the helping vowel is ``i``: ``yi-zurʕ``), so
    ``ya-`` before a person label is always the stem's ``a`` cut off. Moves
    the letter only; the glosses stay aligned. Applied to what the parser
    proposes, not to saved glosses.
    """
    if not segmentation or not gloss:
        return segmentation
    match = _IV_PERSON_PREFIX.match(segmentation)
    if not match or not _PERSON_LABEL.match(re.split(r"[-=.]", gloss, maxsplit=1)[0]):
        return segmentation
    return f"{match.group(1)}-a{segmentation[match.end():]}"


def nc_plural_participle(segmentation: str | None, gloss: str | None) -> str | None:
    """The participial plural ``-in`` of the NC carries gender and number, not
    person: ``mʕany-in sing.NC-MP``, not ``sing.NC-3MP``.

    Muše Sanžar (author's columns, 29.09.2026): ``NC-MP`` 83 times, every one
    on an ``-in`` ending; the six ``NC-3mp`` the author kept are all on
    ``-un`` (``ṯyill-un``, ``Zlill-un``), which this leaves alone, as it
    leaves ``SC``/``PC`` (see ``sc_plural_person``). The published IGT the A1
    tier reads writes ``NC-3MP`` on ``-in``; the parser repeats it with HIGH
    confidence. Applied to what the parser proposes, not to saved glosses.
    """
    if not segmentation or not gloss:
        return gloss
    forms = re.split(r"[-=+]", segmentation)
    labels = re.split(r"[-=+]", gloss)
    if len(forms) != len(labels):
        return gloss
    separators = re.findall(r"[-=+]", gloss)
    changed = False
    for index in range(1, len(labels)):
        previous = labels[index - 1].upper()
        if not re.search(r"\.NC[12]?$", previous):
            continue
        match = _NC_PERSON.match(labels[index])
        if match and _PARTICIPLE_PLURAL.search(forms[index]):
            labels[index] = match.group(1)
            changed = True
    if not changed:
        return gloss
    out = labels[0]
    for separator, label in zip(separators, labels[1:]):
        out += separator + label
    return out


def _alias_map(aliases: Mapping[str, str] | None = None) -> dict[str, str]:
    merged = dict(DEFAULT_GLOSS_ALIASES)
    for source, target in (aliases or {}).items():
        merged[str(source).translate(LOOKALIKE_TRANSLATION).strip().upper()] = (
            str(target).translate(LOOKALIKE_TRANSLATION).strip().upper()
        )
    return merged


def canonicalize_gloss_label(
    component: str,
    known_labels: set[str] | None = None,
    aliases: Mapping[str, str] | None = None,
) -> str | None:
    """Return the canonical grammatical label, or ``None`` for lexical text.

    Aliases are resolved case-insensitively and transitively.  A cycle is
    rejected by returning ``None`` rather than silently producing unstable data.
    """
    cleaned = component.translate(LOOKALIKE_TRANSLATION).strip()
    if not cleaned:
        return None
    known = (AUTHORITATIVE_CONJUGATION_LABELS
             | {item.upper() for item in (known_labels or set())})
    alias_map = _alias_map(aliases)
    value = cleaned.upper()
    seen: set[str] = set()
    while value in alias_map:
        if value in seen:
            return None
        seen.add(value)
        value = alias_map[value]
    if value in known or PERSON_FEATURE_RE.fullmatch(value) is not None:
        return value
    # Alias targets are canonical by definition even if a caller supplied a
    # deliberately minimal known-label set.
    if cleaned.upper() in alias_map:
        return value
    return None


def is_grammatical_label(
    component: str,
    known_labels: set[str] | None = None,
    aliases: Mapping[str, str] | None = None,
) -> bool:
    """Return True for a confirmed label, alias, or person-feature bundle."""
    return canonicalize_gloss_label(component, known_labels, aliases) is not None


def normalize_gloss_with_trace(
    gloss: str,
    known_labels: set[str] | None = None,
    aliases: Mapping[str, str] | None = None,
) -> tuple[str, list[dict[str, str]]]:
    """Canonicalize grammatical labels while preserving lexical glosses.

    The returned trace records deterministic alias replacements, e.g.
    ``nc1 -> PRF``.  This allows the pipeline to preserve the source IGT and
    explain why the canonical output differs from it.
    """
    cleaned = gloss.translate(LOOKALIKE_TRANSLATION).strip()
    parts = SEPARATOR_RE.split(cleaned)
    output: list[str] = []
    replacements: list[dict[str, str]] = []
    for index, part in enumerate(parts):
        if part in {"-", "=", "."}:
            output.append(part)
            continue
        stripped = part.strip()
        uncertainty = ""
        uncertainty_match = re.fullmatch(r"(.+?)(\?+)", stripped)
        label_candidate = stripped
        if uncertainty_match:
            label_candidate = uncertainty_match.group(1)
            uncertainty = uncertainty_match.group(2)
        canonical = canonicalize_gloss_label(label_candidate, known_labels, aliases)
        if (canonical == "DO" and index + 1 < len(parts) and parts[index + 1] == "."
                and (index == 0 or parts[index - 1] in {"-", "="})):
            # the verb `do` at the head of its cell (`do.nc-mp`), not the
            # label DO: see `lexical_do`
            canonical = None
            label_candidate = label_candidate.lower()
        if canonical is not None:
            output.append(canonical + uncertainty)
            source_upper = label_candidate.upper()
            if source_upper != canonical:
                replacements.append({
                    "source": label_candidate,
                    "source_label": source_upper,
                    "target": canonical,
                })
        else:
            # Preserve unknown components exactly.  An all-caps component in an
            # IGT line may be an unsupported grammatical label and must remain
            # visible to the discrepancy engine rather than being silently
            # converted into a lexical gloss.
            output.append(label_candidate + uncertainty)
    return "".join(output), replacements


def normalize_gloss(
    gloss: str,
    known_labels: set[str] | None = None,
    aliases: Mapping[str, str] | None = None,
) -> str:
    """Return the canonical gloss string used in analyses and exports."""
    return normalize_gloss_with_trace(gloss, known_labels, aliases)[0]
