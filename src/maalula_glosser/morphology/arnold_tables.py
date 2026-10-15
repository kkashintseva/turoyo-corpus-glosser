"""Paradigm tables transcribed from Arnold's Grammatik — draft, not in use.

The tables live in ``data/grammar_kb/arnold_tables/*.md`` as Markdown, because
they exist to be corrected by hand: the forms were read off the PDF's OCR
layer, which drops every diacritic, and only their skeletons could be checked
against data the project already trusts.

`readings()` is what the analyzer consults — bare surface to segmentation and
gloss — and `ENABLED` is the switch that lets it. An exact cell fills a gap the
corpus leaves and never outranks it; see `plan_activation` for what is done and
what is not. `load()` returns the tables themselves, for looking at.

Format of a table file
----------------------
A `##` heading opens a table and names it:

    ## SBJV — rdy "plough" (Arnold Gr. S. 145)

Then a Markdown table with a fixed header:

    | p | g | sg | gloss.sg | pl | gloss.pl | ok |

`p` is 1/2/3, `g` is m/f/c. A cell is a form in the project's orthography, `/`
separating Arnold's variants, `(x)` an optional segment, `-` and `=` the
morpheme boundaries. An empty cell means Arnold prints none.

`ok` is written by `scripts/check_arnold_tables.py`, never by hand:

    ✓  the exact form occurs in the project's own data
    ≈  only the skeleton occurs — the diacritics are my reconstruction
    ?  nothing comparable in the project's data; OCR alone

`≈` and `?` are the cells worth your time.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

# The analyzer consults the tables when this is on. Off, `readings()` still
# parses — the checker needs it — but nothing reaches a gloss.
ENABLED = True

TABLES = Path("data/grammar_kb/arnold_tables")

HEADER = ("p", "g", "sg", "gloss.sg", "pl", "gloss.pl", "ok")
CONFIRMED, SKELETON, UNCHECKED = "✓", "≈", "?"

_HEADING = re.compile(r"^##\s+(?P<category>[A-Z][A-Z.]*)\s+—\s+(?P<lexeme>.+?)\s*$")
_SOURCE = re.compile(r"\(Arnold Gr\. S\. (\d+)\)")


@dataclass
class Cell:
    """One paradigm cell: a form, its gloss, and how well it is checked."""
    person: str
    gender: str
    number: str
    form: str
    gloss: str
    status: str = UNCHECKED


@dataclass
class Table:
    category: str          # PST, SBJV, IPV, PRS, PRF, ...
    lexeme: str            # `rdy "plough"`
    page: int | None       # printed page in the Grammatik
    path: Path
    cells: list[Cell] = field(default_factory=list)


def skeleton(form: str) -> str:
    """What survives the OCR: letters without diacritics, length or ə.

    Two forms with the same skeleton are the same word as far as the scan can
    tell, which is exactly the comparison the checker needs — and exactly the
    reason these tables cannot be trusted without one.
    """
    text = unicodedata.normalize("NFD", form or "").lower()
    text = "".join(c for c in text if not unicodedata.combining(c))
    for source, target in (("ʕ", "c"), ("ʔ", ""), ("ə", ""), ("ð", "d"),
                           ("þ", "t"), ("ġ", "g")):
        text = text.replace(source, target)
    return re.sub(r"[^a-z]", "", text)


# A grammatical label is written upper case — except a full person + gender +
# number bundle, which is written lower case: `3fs`, `2mp`. The inventory states
# the rule and holds the list (`case_map`, `_case_rule` in gloss_inventory.json);
# anything short of all three keeps its capitals, so `1S`, `MP`, `3M` and `F` do.
#
# The lexical gloss keeps its own spelling: šwy is glossed `do`, the verb, and
# must not become `DO`, the object marker.
_TAM = {"pst", "prs", "prf", "sbjv", "ipv", "infc", "prfc", "sc", "nc", "pc"}
_STRUCTURAL = {"intf", "neg", "hd", "free", "cns", "indf", "def", "exist",
               "dom", "num", "aff", "emp", "prsv", "recp", "refl"}
_PNG = re.compile(r"^([123][mfc]?[sp]?|[mfc][sp]?|[sp])$", re.I)
_FULL_BUNDLE = re.compile(r"^[123][MF][SP]$")

# The IGT label a stem's TAM is written with. `CAT_TO_TAM` in compositional.py
# holds the same correspondence the other way round — SC↔PST, NC↔PRS, PC↔SBJV —
# because PST/PRS/SBJV are the internal TAM values and SC/NC/PC are what a gloss
# says.
#
# PST is always SC, except on the copula: `w-ōb`, `w-ayb-in` `PST-be-MP`, where
# PST is a morpheme of its own rather than a stem's TAM. That is why the
# inventory's display_map folds PRS and SBJV and carries no PST entry at all —
# a blanket fold would destroy the copular marker. The discriminator is
# positional and the corpus is unanimous about it: of the 94 glosses that write
# PST between boundaries of its own, every one is on `be` or `EXIST`, while a
# PST after a `.` is a stem's, as in `come.PST`.
_IGT_LABEL = {"PST": "SC", "PRS": "NC", "SBJV": "PC"}


def normalise_gloss(gloss: str) -> str:
    """Write a gloss's labels the way IGT writes them, leave the word alone."""
    out: list[str] = []
    for part in re.split(r"([-=./ ])", gloss or ""):
        low = part.lower()
        if low in _TAM or low in _STRUCTURAL or _PNG.match(part):
            part = part.upper()
            if out and out[-1] == ".":
                part = _IGT_LABEL.get(part, part)
            if _FULL_BUNDLE.match(part):
                part = part.lower()
        out.append(part)
    return "".join(out)


def _split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def load(root: Path | str = ".") -> list[Table]:
    """Every table on disk, in file order. Malformed rows are skipped."""
    directory = Path(root) / TABLES
    tables: list[Table] = []
    for path in sorted(directory.glob("*.md")):
        current: Table | None = None
        for line in path.read_text(encoding="utf-8").splitlines():
            heading = _HEADING.match(line)
            if heading:
                page = _SOURCE.search(heading["lexeme"])
                current = Table(category=heading["category"],
                                lexeme=_SOURCE.sub("", heading["lexeme"]).strip(),
                                page=int(page[1]) if page else None, path=path)
                tables.append(current)
                continue
            if current is None or not line.startswith("|"):
                continue
            row = _split_row(line)
            if len(row) != len(HEADER) or row[0] in ("p", "---") or set(row[0]) <= {"-"}:
                continue
            for number, form, gloss in (("s", row[2], row[3]), ("p", row[4], row[5])):
                if form:
                    current.cells.append(Cell(person=row[0], gender=row[1],
                                              number=number, form=form,
                                              gloss=gloss, status=row[6] or UNCHECKED))
    return tables


def variants(form: str) -> list[str]:
    """`rdā/irda` and `ḥamm(i)` spell more than one surface."""
    out: list[str] = []
    for piece in form.split("/"):
        piece = piece.strip()
        if not piece:
            continue
        bare = re.sub(r"\([^)]*\)", "", piece)
        full = piece.replace("(", "").replace(")", "")
        out.extend(dict.fromkeys(x for x in (full, bare) if x))
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# Using them
# ---------------------------------------------------------------------------
def readings(root: Path | str = ".") -> dict[str, list[tuple[str, str, str, str]]]:
    """Bare surface → [(segmentation, gloss, status, source)].

    A cell can spell more than one word: `rḏā/irḏa` is two, `ḥamm(i)` is two,
    and where the gloss column carries the same number of `/` variants they are
    paired with the forms one for one — `ḥim-ən/ḥim-n=i` against
    `see.SC-INTF.1S/see.SC-INTF=1S`. A single gloss covers every variant.
    """
    out: dict[str, list[tuple[str, str, str, str]]] = {}
    for table in load(root):
        where = f"{table.path.name} {table.category} S. {table.page}"
        for cell in table.cells:
            # `-xun`, `-(h)en`: an exponent, not a word. The pronominal suffix
            # table is an inventory of bound forms and has no business in a
            # lookup keyed by surface.
            if cell.form.startswith(("-", "=")):
                continue
            forms, glosses = variants(cell.form), [
                g.strip() for g in cell.gloss.split("/") if g.strip()]
            if not forms or not glosses:
                continue
            for index, segmentation in enumerate(forms):
                gloss = glosses[index] if len(glosses) == len(forms) else glosses[0]
                surface = segmentation.replace("-", "").replace("=", "")
                if not surface:
                    continue
                entry = (segmentation, gloss, cell.status, where)
                bucket = out.setdefault(surface, [])
                if entry not in bucket:
                    bucket.append(entry)
    for surface, entries in menn_pattern_readings(root).items():
        # a printed cell for the same surface wins: the rule only fills gaps
        if surface not in out:
            out[surface] = entries
    return out


# Arnold § 8.2.3, S. 390, after the menn- table: "Der Bildungsweise von menn-
# schließen sich folgende Präpositionen an: ʕemm- mit, ḓett- gegen; außerdem,
# jedoch ohne Umlautung von Vokalen: ġapp- bei, kuḥkull- um herum, ʕayatt-
# wegen, luḳᵊbl- in Richtung auf; mit Umlautung von o zu u in vortoniger Silbe
# sowie bei Antritt von Suffixen der 2 sg f und 1 sg c: ḳomm- vor, roḥl-
# hinter; und mit Kürzung des Langvokals ū in vortoniger Silbe: ḳūr- neben."
#
# A rule Arnold states for a closed list, not a class induced from a table, so
# it lives next to the tables and yields the same kind of reading. The menn-
# cells it copies (S. 390): menn=e menn=a menn=ax minn=iš minn(=i); minnay=hun
# minnay=y/minnay=n minnay=hen minnay=xun minnay=xen minnay=naḥ. Bases are in
# the project spelling (ḳ→k, k→c, ḓ→ḏ̣: ḏ̣ett-, cuḥcull-, komm-, kūr-), as the
# lexicon has them. luḳᵊbl- is left out: where its ə goes before a suffix is
# not shown. Arnold's dictionary agrees where it prints forms (ḏ̣ette,
# ġappe/ġappa/ġappax/ġappiš/ġapp/ġappayy/ġappayxun/ġappaynaḥ, ʕayatte, komme,
# kumm/kummi/kummayy/kummayxun, roḥle/roḥla/ruḥəl/ruḥlayhun/ruḥlaynaḥ) and
# also prints plurals the rule does not make (ġappēn, ġappāy, kummēn,
# kummāy, cuḥcullinnun): those are not generated here.
#
# The meaning is the author's (closed class, from Glossing_updated.docx: ʕemm
# with, ḏ̣ett against, ġapp own, ʕayatt because, komm before, kūr near) where
# the closed class has the base, else the tuple's. A form on a host the closed
# class already composes (ʕemm=e, kummay=xun, ġappay=naḥ) is left to it; the
# rule adds the allomorphs it lacks: ʕimm-, ʕimmay-/ʕemmay-, ḏ̣itt-, ḏ̣ittay-,
# ʕayattay-, kuray-, roḥl-/ruḥl-/ruḥlay-, cuḥcull-/cuḥcullay-.
#
# (stressed base, base before 2 sg f and 1 sg c, bare 1 sg or "" where the
# bare base is the free preposition (cuḥcull šimša "um die Sonne herum") or
# is not attested, pretonic plural bases, gloss)
MENN_PATTERN: tuple[tuple[str, str, str, tuple[str, ...], str], ...] = (
    # the author writes the plural both ways: ʕemmáy and ʕimmáy (Muše Sanžar)
    ("ʕemm", "ʕimm", "ʕimm", ("ʕimm", "ʕemm"), "with"),
    ("ḏ̣ett", "ḏ̣itt", "ḏ̣itt", ("ḏ̣itt",), "against"),
    ("ġapp", "ġapp", "ġapp", ("ġapp",), "at"),
    ("cuḥcull", "cuḥcull", "", ("cuḥcull",), "around"),
    ("ʕayatt", "ʕayatt", "", ("ʕayatt",), "because.of"),
    ("komm", "kumm", "kumm", ("kumm",), "in.front.of"),
    # the bare 1 sg needs its helper vowel: ruḥəl (Wb., III 30.59)
    ("roḥl", "ruḥl", "ruḥəl", ("ruḥl",), "behind"),
    # "neben, in der Nähe von"; the author glosses kurayy kūray=y near=3P
    ("kūr", "kūr", "", ("kur",), "near"),
)
MENN_RULE = "§ 8.2.3 S. 390 (menn- rule)"


def _closed_class_prepositions(root: Path | str) -> tuple[dict[str, str], set[str]]:
    """Host form → meaning, and every preposition form the closed class has."""
    import json
    hosts: dict[str, str] = {}
    forms: set[str] = set()
    path = Path(root) / "data" / "grammar_kb" / "closed_class_v2.jsonl"
    if not path.is_file():
        return hosts, forms
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("category") not in ("preposition", "pseudoverb"):
            continue
        form = unicodedata.normalize("NFC", entry.get("form") or "")
        forms.add(form)
        if entry.get("attachment") == "host":
            hosts.setdefault(form, (entry.get("gloss") or "").rstrip("="))
    return hosts, forms


def menn_pattern_readings(root: Path | str = ".") -> dict[str, list[tuple[str, str, str, str]]]:
    """Surface → [(segmentation, gloss, status, source)] for MENN_PATTERN."""
    out: dict[str, list[tuple[str, str, str, str]]] = {}
    hosts, forms = _closed_class_prepositions(root)

    def add(segmentation: str, gloss: str) -> None:
        surface = segmentation.replace("=", "")
        if segmentation.split("=")[0] in hosts or surface in forms:
            return                      # the closed class composes it already
        entry = (segmentation, gloss, UNCHECKED, MENN_RULE)
        bucket = out.setdefault(surface, [])
        if entry not in bucket:
            bucket.append(entry)

    for stressed, umlauted, bare, plurals, meaning in MENN_PATTERN:
        meaning = hosts.get(stressed) or meaning
        add(f"{stressed}=e", f"{meaning}=3ms")
        add(f"{stressed}=a", f"{meaning}=3fs")
        add(f"{stressed}=ax", f"{meaning}=2ms")
        add(f"{umlauted}=iš", f"{meaning}=2fs")
        if bare:
            add(bare, f"{meaning}.1S")
        add(f"{umlauted}=i", f"{meaning}=1S")
        for base in plurals:
            add(f"{base}ay=hun", f"{meaning}=3mp")
            add(f"{base}ay=y", f"{meaning}=3P")
            add(f"{base}ay=n", f"{meaning}=3P")
            add(f"{base}ay=hen", f"{meaning}=3fp")
            add(f"{base}ay=xun", f"{meaning}=2mp")
            add(f"{base}ay=xen", f"{meaning}=2fp")
            add(f"{base}ay=naḥ", f"{meaning}=1P")
    return out


# Where the tables sit among the other sources. B1, like the normative v4
# paradigm cells and for the same reason: an exact cell fills a gap and is
# offered for review, but the corpus outranks it. The score is a shade lower
# still where the cell's own mark says the project's data could not confirm it.
TIER = "B1"
SCORE = {CONFIRMED: 0.80, SKELETON: 0.74, UNCHECKED: 0.72}


def plan_activation() -> list[str]:
    """What is done and what is left, in the order it is worth doing.

    1. As exact cells, like `morphology_v4` — done. `readings()` is keyed by
       surface and `CompositionalAnalyzer._from_arnold_tables` consults it in
       the direct-lookup stage. Cheap, safe, and limited to the forms actually
       printed: `yi-rḏ-un` is glossed, `yi-šm-un` is not.

    2. As a class template. Read the exponents off the table (`-un` MP, `-an`
       FP, `-ya` F, ...) together with the stem alternation the table exhibits
       (`rōḏ` ~ `rōḏy`, `iḥəm` ~ `iḥm`), and let the v4 engine inflect any root
       the lexicon marks IIIy. This is what would actually close the gap —
       `lexeme_profiles` admits strong triliterals only — and it is also what
       most needs the tables to be right, because it generalises them.

    3. As a check on what is already there. Every cell marked `✓` is a form two
       independent sources agree on; a cell that contradicts an existing KB row
       is a bug in one of them. `scripts/check_arnold_tables.py` reports the
       marks; the contradictions are not yet looked for.
    """
    return ["exact cells (like morphology_v4) — done",
            "class template for the v4 engine",
            "cross-check against the existing KB"]
