"""LexemeMorphProfile: the per-lexeme record the generator inflects from.

Kept minimal — only the fields the current generator needs. Built from the
exact paradigm cells (verbs), a curated tarba noun, and corpus-attested noun /
adjective stems. Unresolved links go to a conflicts file, not into a guess.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path

from .types import StemSlot, POS


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value or "")


@dataclass(frozen=True)
class LexemeMorphProfile:
    profile_id: str
    lemma: str
    pos: str
    lexeme_id: str | None = None
    pos_status: str = "INFERRED"
    root: tuple[str, ...] | None = None
    root_class: str | None = None
    derivational_stem: str | None = None
    inflection_class: str | None = None
    gender: str | None = None
    stem_slots: tuple[StemSlot, ...] = ()
    paradigm_cell_ids: tuple[str, ...] = ()
    allowed_morphophonology: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()

    def slot(self, name: str) -> str | None:
        for s in self.stem_slots:
            if s.name == name:
                return s.stem
        return None

    def slot_variants(self, name: str) -> tuple[str, ...]:
        """All explicit allomorphs for a slot, preserving source order."""
        return tuple(dict.fromkeys(
            slot.stem for slot in self.stem_slots
            if slot.name == name and slot.stem))

    def as_dict(self) -> dict:
        d = asdict(self)
        d["stem_slots"] = [s.as_dict() for s in self.stem_slots]
        return d


# --- curated irregular / high-value profiles ----------------------------

def _tarba_profile() -> LexemeMorphProfile:
    return LexemeMorphProfile(
        profile_id="prof_tarba", lemma="tarba", pos=POS.NOUN.value,
        pos_status="EXPLICIT", root=("t", "r", "b"),
        inflection_class="NOUN_TARB_CLASS", gender="M",
        stem_slots=(StemSlot("SG_FREE", "tarb"), StemSlot("MP_FREE", "tarb"),
                    StemSlot("MP_OBLIQUE", "tarbōy")),
        allowed_morphophonology=("EPENTHESIS_I", "DEVOICE_B_TO_P"),
        evidence=("PARADIGMEN tarba paradigm",))


def _curated_profiles(root: Path) -> list[LexemeMorphProfile]:
    """Load explicitly sourced lexeme profiles that are absent from Arnold.

    The JSONL is an append-only provenance layer, not a surface correction:
    each row names a lexeme, class and licensed stem slots for the ordinary
    generator/analyzer machinery.
    """
    path = root / "data/grammar_kb/morphology_v4/curated_lexeme_profiles.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["root"] = tuple(row.get("root") or ()) or None
        row["stem_slots"] = tuple(
            StemSlot(**slot) for slot in row.get("stem_slots") or ())
        for name in ("paradigm_cell_ids", "allowed_morphophonology",
                     "evidence", "conflicts"):
            row[name] = tuple(row.get(name) or ())
        out.append(LexemeMorphProfile(**row))
    return out


# --- corpus-derived noun / adjective stems ------------------------------

FREE_RE = re.compile(r"-a$")


def _noun_profiles_from_corpus(root: Path) -> tuple[list[LexemeMorphProfile], list[dict]]:
    """A noun/adjective profile per corpus lemma with a clear stem.

    A token segmented `X-a` glossed `word-FREE` gives a SG_FREE stem X; `X-ō`
    glossed `word-MP` (or `-P`) gives the plural stem; an adjective is a lemma
    seen with `-in`/`-an`.
    """
    profiles: dict[str, dict] = defaultdict(lambda: {"slots": {}, "pos": None,
                                                     "gender": None, "ev": set()})
    # TRAIN ONLY. Deriving stems from dev/test would leak held-out morphology
    # into the profiles the analyzer inflects from; only train examples,
    # Arnold, MASC and the normative paradigms are permitted sources.
    train_ids = {json.loads(l)["example_id"]
                 for l in (root / "data/evaluation/split_examples.jsonl")
                 .read_text().splitlines() if l.strip()
                 and json.loads(l).get("split") == "train"}
    for line in (root / "data/final/igt_examples.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        example = json.loads(line)
        if example.get("example_id") not in train_ids:
            continue
        for tok in example.get("tokens") or []:
            seg = nfc(tok.get("segmentation") or "")
            gloss = nfc(tok.get("gloss") or "")
            if "=" in seg or "=" in gloss:
                continue                      # skip cliticised for the stem set
            parts = seg.split("-")
            gparts = gloss.split("-")
            if len(parts) != len(gparts) or len(parts) < 2:
                continue
            stem, lastseg = parts[0], parts[-1]
            lastg = gparts[-1]
            lemma = gparts[0]
            if not lemma or lemma.isupper():
                continue
            p = profiles[lemma]
            p["ev"].add(tok.get("surface_canonical"))
            if lastg == "FREE" and lastseg == "a":
                p["slots"]["SG_FREE"] = stem
                p["pos"] = p["pos"] or "NOUN"
                if "-č-" in seg or "-ṯ-" in seg or "-t-" in seg:
                    p["gender"] = "F"
            elif lastg in ("P", "MP") and lastseg in ("ō", "o"):
                p["slots"]["MP_FREE"] = stem
                p["pos"] = p["pos"] or "NOUN"
            elif lastg.endswith("MP.INDF") or lastseg == "in":
                p["slots"].setdefault("MS", stem)
                p["pos"] = "ADJ"
            elif lastg.endswith("FP.INDF") or lastseg == "an":
                p["slots"].setdefault("MS", stem)
                p["pos"] = "ADJ"
    out, conflicts = [], []
    for lemma, p in profiles.items():
        if not p["slots"]:
            continue
        pid = "prof_" + re.sub(r"[^a-z0-9]", "", lemma.lower())[:24]
        out.append(LexemeMorphProfile(
            profile_id=pid, lemma=lemma, pos=p["pos"] or "NOUN",
            pos_status="DERIVED", gender=p["gender"],
            stem_slots=tuple(StemSlot(n, s, "DERIVED") for n, s in sorted(p["slots"].items())),
            allowed_morphophonology=("EPENTHESIS_I", "DEVOICE_B_TO_P"),
            evidence=("corpus",)))
    return out, conflicts


def _verb_profiles_from_cells(root: Path) -> list[LexemeMorphProfile]:
    cells = [json.loads(l) for l in
             (root / "data/grammar_kb/morphology_v4/paradigm_cells.jsonl")
             .read_text().splitlines() if l.strip()]
    by_lex: dict[str, list[dict]] = defaultdict(list)
    for c in cells:
        if c["source"] == "paradigms_final" and c.get("lexeme_heading"):
            by_lex[c["lexeme_heading"]].append(c)
    out = []
    for lex, group in by_lex.items():
        out.append(LexemeMorphProfile(
            profile_id="prof_" + re.sub(r"[^a-zāēīōū]", "", lex.lower())[:24],
            lemma=lex, pos=POS.VERB.value, pos_status="EXPLICIT",
            inflection_class="IRREGULAR_" + lex,
            paradigm_cell_ids=tuple(c["cell_id"] for c in group),
            evidence=("paradigms_final.docx",)))
    return out


CONS = "bcčdḏfgġhḥjklmnpqrsṣštṭṯvwxyzžʔʕḳ"


def _consonants(surface: str) -> str:
    return "".join(ch for ch in surface if ch in CONS)


SBJV_PREFIXES = ("yi", "či", "ni", "y", "č", "n")
# PST person suffixes (Glossing labels -> exponent), the suffixed stem precedes
PST_SUFFIXES = {"3FS": "aṯ", "1S": "iṯ", "2MS": "ič", "2FS": "iš",
                "2MP": "ičxun", "2FP": "ičxen", "1P": "innaḥ"}
# PRS/PRF oblique suffixes on the oblique stem (MS is the bare stem)
OBLIQUE_SUFFIXES = {"F": "a", "3FS": "a", "MP": "in", "FP": "an"}


def _png_from_label(label: str) -> str:
    """The trailing PNG bundle of a gloss like open.PST-3FS -> 3FS."""
    m = re.search(r"[-.]([123]?(?:M|F|C)?(?:S|P)|MP|FP|MS|FS)$", label)
    return m.group(1) if m else ""


def _lemma_and_person(gloss: str, tam: str) -> tuple[str, bool]:
    """The lexical lemma (before .TAM) and whether a person prefix is present.
    '3F-be.hit.SBJV' -> ('be.hit', True); 'go.up.PST-1S' -> ('go.up', False)."""
    before = re.split(rf"\.{tam}\b", gloss, maxsplit=1)[0]
    m = re.match(r"^([123])(?:M|F|C)?-(.+)$", before)
    return (m.group(2), True) if m else (before, False)


def _extract_verb_slots(seg: str, gloss: str, surface, by_lemma) -> None:
    """Route one attested verb form to the right stem slot of its lexeme.

    Only clean cases: a person-prefixed SBJV, a bare PST/PRS/PRF citation, or a
    suffixed PST/PRS/PRF (stem + a recognised PNG exponent). The stem is what
    precedes the exponent; the exponent itself is never stored.
    """
    tam = next((t for t in ("SBJV", "PST", "PRS", "PRF") if f".{t}" in gloss), None)
    if not tam:
        return
    lemma, has_person = _lemma_and_person(gloss, tam)
    if not lemma:
        return
    p = by_lemma[lemma]

    # SBJV: person prefix + bare stem (no object suffix)
    if tam == "SBJV" and has_person and "-" in seg:
        prefix, rest = seg.split("-", 1)
        if prefix in SBJV_PREFIXES and "-" not in rest:
            p["slots"]["SBJV_STEM"] = rest
            p["root"] = _consonants(rest)
            p["ev"].add(surface)
        return

    png = _png_from_label(gloss)
    if "-" not in seg:
        # a bare citation (3MS): PST citation must look like i C C V C
        if not has_person and 3 <= len(seg) <= 9 and not png:
            slot = "PST_CITATION" if tam == "PST" else f"{tam}_MS"
            if tam == "PST" and seg[:1] not in "iīe":
                return
            p["slots"].setdefault(slot, seg)
            p["root"] = p["root"] or _consonants(seg)
            p["ev"].add(surface)
        return

    if not png:
        return
    stem, exp = seg.rsplit("-", 1)
    if "-" in stem or has_person:                 # single suffix, no person prefix
        return
    if tam == "PST" and PST_SUFFIXES.get(png) == exp:
        p["slots"].setdefault("PST_SUFFIXED", stem)
    elif tam in ("PRS", "PRF") and OBLIQUE_SUFFIXES.get(png) == exp:
        p["slots"].setdefault(f"{tam}_OBLIQUE", stem)
    else:
        return
    p["root"] = p["root"] or _consonants(stem)
    p["ev"].add(surface)


def _strong_g_profiles_from_train(root: Path, train_ids: set) -> tuple[list[LexemeMorphProfile], list[dict]]:
    """Strong-G verb stems from TRAIN analyses (an allowed, attested source).

    A train token segmented `yi-nfuḏ` glossed `3M-arrive.SBJV` gives the SBJV
    stem `nfuḏ` for lexeme `arrive`; a bare PST like `iftaḥ` glossed `open.PST`
    gives the PST citation. From one attested form the generator can then
    produce the rest of that lexeme's paradigm (a genuinely unseen inflection
    of a known lexeme, not a held-out lookup). Keyed by the lexical gloss so
    homonyms with different roots are not merged.
    """
    by_lemma: dict[str, dict] = defaultdict(lambda: {"slots": {}, "root": None, "ev": set()})
    for line in (root / "data/final/igt_examples.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        ex = json.loads(line)
        if ex.get("example_id") not in train_ids:
            continue
        for tok in ex.get("tokens") or []:
            seg = nfc(tok.get("segmentation") or "")
            gloss = nfc(tok.get("gloss") or "")
            if "=" in seg or "=" in gloss:
                continue
            _extract_verb_slots(seg, gloss, tok.get("surface_canonical"), by_lemma)
    out, conflicts = [], []
    for lemma, p in by_lemma.items():
        slots = p["slots"]
        if "SBJV_STEM" not in slots and "PST_CITATION" not in slots:
            continue
        # STRONG_G is the strong triliteral class. A stem with fewer than three
        # root consonants (šwy -> šu) is weak/biliteral, a different class: keep
        # it out of STRONG_G rather than patch it with a surface condition.
        stem = slots.get("SBJV_STEM") or slots.get("PST_CITATION") or ""
        if len(_consonants(stem)) < 3:
            conflicts.append({"lemma": lemma, "stem": stem,
                              "reason": "fewer than three root consonants; "
                                        "not a strong triliteral", "excluded": True})
            continue
        pid = "prof_sg_" + re.sub(r"[^a-zāēīōū]", "", lemma.lower())[:20]
        out.append(LexemeMorphProfile(
            profile_id=pid, lemma=lemma, pos=POS.VERB.value, pos_status="DERIVED",
            root=tuple(p["root"]) if p["root"] else None, root_class="STRONG_G",
            derivational_stem="G", inflection_class="STRONG_G",
            stem_slots=tuple(StemSlot(n, s, "DERIVED") for n, s in sorted(slots.items())),
            allowed_morphophonology=("EPENTHESIS_I",),
            evidence=("train",)))
    return out, conflicts


def _arnold_verb_slots(root: Path) -> tuple[dict, list[dict]]:
    """PRS/PRF oblique stems from Arnold's already-mined Maalula forms.

    Reuses data/dictionary_morphology_v1/inflected_forms.jsonl (Arnold, not
    held-out). Only a strong triliteral oblique form (stem + -in MP / -an FP /
    -a FS) whose stem consonants match the recorded 3-consonant root is taken;
    the exponent is stripped, the stem stored. Keyed by root so it can merge
    into a train strong-G profile of the same root. External source: an Arnold
    win must be a productively generated PNG, not this exact form.
    """
    path = root / "data/dictionary_morphology_v1/inflected_forms.jsonl"
    if not path.exists():
        return {}, []
    by_root: dict[str, dict] = defaultdict(lambda: {"slots": {}, "ev": []})
    conflicts = []
    OBL = {"in": "MP", "an": "FP"}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if not (r.get("selected") and r.get("usable_for_maalula")):
            continue
        tam = r.get("tam_canonical")
        if tam not in ("PRS", "PRF") or r.get("effective_dialect") not in ("M", None):
            continue
        surf = nfc(r.get("surface_canonical") or "")
        root_cons = _consonants(nfc(r.get("root") or ""))
        if not surf or len(root_cons) != 3 or "-" in surf or "=" in surf or "(" in surf:
            continue
        for exp, _png in OBL.items():
            if surf.endswith(exp) and len(surf) - len(exp) >= 3:
                stem = surf[: -len(exp)]
                if _consonants(stem) != root_cons:
                    continue
                slot = f"{tam}_OBLIQUE"
                d = by_root[root_cons]
                if slot in d["slots"] and d["slots"][slot] != stem:
                    conflicts.append({"root": root_cons, "slot": slot,
                                      "have": d["slots"][slot], "new": stem})
                    continue
                d["slots"].setdefault(slot, stem)
                d["ev"].append({"page": r.get("source_page"),
                                "form": surf, "tam": tam})
                break
    return by_root, conflicts


# Arnold writes ǧ/ć where the corpus surfaces use ž/č; normalise the mined
# stems to the corpus convention so they reconstruct real surfaces (miǧćamʕ ->
# mižčamʕ, the VIII present participle of žmʕ 'gather').
_ARNOLD_TRANSCRIPTION = str.maketrans("ǧć", "žč")

DERIVED_STEM_NAMES = {
    "II": "Paʕʕel",
    "III": "III",
    "IV": "ʔApʕel",
    "I7": "VII",
    "I8": "VIII",
    "I10": "I10",
    # Arnold's ``2`` classes are the reflexive/passive counterparts of the
    # corresponding derived stems.  Keep the source labels distinct: merging
    # them into II/III/IV would lose a morphologically contrastive č(i)- stem.
    "II2": "II2",
    "III2": "III2",
    "IV2": "IV2",
}
REFLEXIVE_DERIVED_CLASSES = frozenset({"II2", "III2", "IV2"})
DERIVED_SLOT_NAMES = (
    "PST_CITATION", "PST_SUFFIXED", "SBJV_STEM", "PRS_MS",
    "PRS_OBLIQUE", "PRF_MS", "PRF_OBLIQUE", "IPV_STEM",
)


def _iv2_citation(value: str) -> str:
    """Normalize the IV₂ principal-part prefix from Arnold's extraction."""
    value = nfc(value or "").translate(_ARNOLD_TRANSCRIPTION)
    value = value.replace("́", "")
    return value[2:] if value.startswith("yi") else value


def _iv2_oblique_variants(stem: str) -> tuple[str, ...]:
    """The local IV₂ table licenses zero/ə/i epenthesis in the oblique stem."""
    base = _syncopate_theme(stem)
    if len(base) < 3:
        return (base,)
    index = len(base) - 2
    return tuple(dict.fromkeys((
        base,
        base[:index] + "ə" + base[index:],
        base[:index] + "i" + base[index:],
    )))


def _iv2_prf_stems(citation: str) -> tuple[str, str]:
    """IV₂ PRF 3SG.M and non-SG oblique stems (§2.10.5)."""
    if citation.endswith("aḥ"):
        return citation[:-2] + "eḥ", citation[:-2] + "īḥ"
    if citation.endswith("aḳ"):
        return citation[:-2] + "eḳ", citation[:-2] + "īḳ"
    if len(citation) >= 2:
        return citation[:-1] + "e" + citation[-1], citation[:-1] + "ī" + citation[-1]
    return citation, citation


def _strip_finite_subject_prefix(surface: str, tam: str,
                                  person: int | None) -> str:
    """Remove an overt NC/PRF subject prefix from an Arnold finite form.

    Third person has no prefix in the nominal conjugations.  PRF may insert
    ``i`` before a consonant cluster (ni-/či-), whereas PRS consistently uses
    n-/č-.  The translation is never consulted.
    """
    if person == 1:
        prefixes = ("ni", "n") if tam == "PRF" else ("n",)
    elif person == 2:
        prefixes = ("či", "č") if tam == "PRF" else ("č",)
    else:
        return surface
    return next((surface[len(prefix):] for prefix in prefixes
                 if surface.startswith(prefix)), surface)


def _pst_subject_ending(person: int | None, number: str | None,
                        gender: str | None) -> str | None:
    """Return the general PST subject ending for an explicit PNG."""
    if person == 1:
        return PST_SUFFIXES["1P" if number == "P" else "1S"]
    if person == 2 and number == "S":
        return PST_SUFFIXES["2FS" if gender == "F" else "2MS"]
    if person == 2 and number == "P":
        return PST_SUFFIXES["2FP" if gender == "F" else "2MP"]
    if person == 3 and number == "S" and gender == "F":
        return PST_SUFFIXES["3FS"]
    return None


def _source_class_overlay(root: Path) -> dict[tuple[str, str], str]:
    """Confirmed source class relabels, kept in a provenance artifact rather than
    edited into roots_and_classes.jsonl (e.g. iččxel is mislabelled I2 but is
    I8). Returns {(entry_id, from_class): to_class}."""
    path = root / "data/grammar_kb/morphology_v4/source_corrections.jsonl"
    out: dict[tuple[str, str], str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") == "class_relabel":
                out[(row["entry_id"], row["from_class"])] = row["to_class"]
    return out


def _derived_verb_profiles(root: Path) -> tuple[list[LexemeMorphProfile], list[dict]]:
    """Profiles from Arnold's explicit class labels and inflected forms.

    ``inflected_forms.jsonl`` is a structured extraction of the local
    ``data/raw/arnold_full.json``.  A slot is admitted only from a Maalula form
    with an explicit TAM/PNG label.  Agreement endings may be stripped to get
    a DERIVED stem, but translations are never consulted.
    """
    overlay = _source_class_overlay(root)
    roots = {}
    for line in (root / "data/dictionary_morphology_v1/roots_and_classes.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for stem in row.get("stems") or []:
            cls = overlay.get((row["entry_id"], stem.get("class")), stem.get("class"))
            if cls in DERIVED_STEM_NAMES:
                roots.setdefault((row["entry_id"], cls), {
                    "root": row.get("root"), "page": row.get("page"),
                    "perfect": stem.get("perfect"), "imperfect": stem.get("imperfect"),
                    "grammar_ref": stem.get("grammar_ref"),
                })

    values: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list))

    def _record_imperative(key, surface: str,
                           number: str | None, gender: str | None) -> None:
        slot = {
            ("S", "M"): "IPV_MS", ("S", "F"): "IPV_FS",
            ("P", "M"): "IPV_MP", ("P", "F"): "IPV_FP",
        }.get((number, gender))
        if not slot:
            return
        # Arnold sometimes prints alternatives in one field (ʕubrun/ʕubrōn).
        # They are genuine allomorphs of one cell, not a conflict.
        for variant in (part.strip() for part in surface.split("/")):
            if variant:
                values[key][slot].append(variant)
                if slot == "IPV_MS":       # backwards-compatible canonical slot
                    values[key]["IPV_STEM"].append(variant)

    for line in (root / "data/dictionary_morphology_v1/inflected_forms.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cls = overlay.get((row.get("source_entry_id"), row.get("arnold_stem_class")),
                          row.get("arnold_stem_class"))
        key = (row.get("source_entry_id"), cls)
        if key not in roots or not (row.get("selected") and row.get("usable_for_maalula")):
            continue
        if row.get("effective_dialect") not in ("M", None):
            continue
        label = row.get("raw_feature_label") or ""
        if "suff." in label or row.get("object_features") or row.get("proclitic"):
            continue
        surface = nfc(row.get("surface_canonical") or "").translate(_ARNOLD_TRANSCRIPTION)
        tam = row.get("tam_canonical")
        features = row.get("features") or {}
        person, number, gender = (features.get("person"), features.get("number"),
                                  features.get("gender"))
        if not surface or "-" in surface or "=" in surface:
            continue
        if cls == "III2":
            if tam == "PST":
                if person == 3 and number == "S" and gender == "M":
                    values[key]["PST_CITATION"].append(surface)
                ending = _pst_subject_ending(person, number, gender)
                if ending and surface.endswith(ending):
                    values[key]["PST_SUFFIXED"].append(surface[:-len(ending)])
            elif tam == "SBJV":
                if person == 1 and surface.startswith("n") \
                        and number in ("S", "P"):
                    # 1SG and 1PL are syncretic and have no number suffix.
                    values[key]["SBJV_STEM"].append(surface[1:])
                elif person == 3 and number == "S" and gender == "M" \
                        and surface.startswith("y"):
                    values[key]["SBJV_STEM"].append(surface[1:])
                elif person in (2, 3) and number == "P":
                    prefix = "č" if person == 2 else "y"
                    ending = "an" if gender == "F" else "un"
                    if surface.startswith(prefix) and surface.endswith(ending):
                        values[key]["SBJV_OBLIQUE"].append(
                            surface[len(prefix):-len(ending)])
            elif tam == "PRS" and number == "S" and gender == "M":
                values[key]["PRS_MS"].append(
                    _strip_finite_subject_prefix(surface, tam, person))
            elif tam == "PRS" and number == "P" and gender == "M" \
                    and surface.endswith("in"):
                values[key]["PRS_OBLIQUE"].append(
                    _strip_finite_subject_prefix(surface[:-2], tam, person))
            elif tam == "PRF" and number == "S" and gender == "M":
                values[key]["PRF_MS"].append(
                    _strip_finite_subject_prefix(surface, tam, person))
            elif tam == "PRF" and number == "P" and gender == "M" \
                    and surface.endswith("in"):
                values[key]["PRF_OBLIQUE"].append(
                    _strip_finite_subject_prefix(surface[:-2], tam, person))
            elif tam == "IPV":
                _record_imperative(key, surface, number, gender)
        elif tam == "PST" and person == 3 and number == "S" and gender == "M":
            values[key]["PST_CITATION"].append(surface)
        elif tam == "PST" and person == 3 and number == "S" and gender == "F" \
                and surface.endswith("aṯ"):
            values[key]["PST_SUFFIXED"].append(surface[:-2])
        elif tam == "SBJV" and person == 3 and number == "S" and gender == "M" \
                and surface.startswith("y"):
            values[key]["SBJV_STEM"].append(surface[1:])
        elif tam == "PRS" and person == 3 and number == "S" and gender == "M":
            values[key]["PRS_MS"].append(surface)
        elif tam == "PRS" and number == "P" and gender == "M" \
                and surface.endswith("in"):
            values[key]["PRS_OBLIQUE"].append(surface[:-2])
        elif tam == "PRF" and person == 3 and number == "S" and gender == "M":
            values[key]["PRF_MS"].append(surface)
        elif tam == "PRF" and number == "P" and gender == "M" \
                and surface.endswith("in"):
            values[key]["PRF_OBLIQUE"].append(surface[:-2])
        elif tam == "IPV":
            _record_imperative(key, surface, number, gender)

    profiles, conflicts = [], []
    for (entry_id, cls), source in roots.items():
        perfect, imperfect = source["perfect"], source["imperfect"]
        if cls in REFLEXIVE_DERIVED_CLASSES:
            perfect = nfc(perfect or "").translate(
                _ARNOLD_TRANSCRIPTION).split("/")[0].strip()
            imperfect = nfc(imperfect or "").translate(
                _ARNOLD_TRANSCRIPTION).split("/")[0].strip()
            # A small, systematic extraction defect copies the y-prefixed
            # imperfect into both principal-part fields.  When the two fields
            # are literally identical, recover only the missing citation
            # prefix boundary (yičšōreṭ -> čšōreṭ); do not repair disagreeing
            # or incomplete principal parts.
            if cls == "III2" and perfect == imperfect and perfect.startswith("yič"):
                perfect = perfect[2:]
        # Two agreeing principal parts are the minimum class-membership gate.
        if not perfect or not imperfect:
            continue
        imp_stem = imperfect[1:] if imperfect.startswith("y") else imperfect
        clean = imp_stem and "." not in imp_stem and " " not in imp_stem \
            and len(_consonants(imp_stem)) >= 2
        slots = []
        slot_names = DERIVED_SLOT_NAMES + (
            ("SBJV_OBLIQUE",)
            if values[(entry_id, cls)].get("SBJV_OBLIQUE") else ())
        for name in slot_names:
            counts = defaultdict(int)
            for value in values[(entry_id, cls)].get(name, []):
                counts[value] += 1
            ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            if ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]):
                status = "EXPLICIT" if name in (
                    "PST_CITATION", "PRS_MS", "PRF_MS", "IPV_STEM") else "DERIVED"
                slots.append(StemSlot(name, ranked[0][0], status))
            else:
                slots.append(StemSlot(name, "", "UNKNOWN" if not ranked else "CONFLICT"))
                if len(ranked) > 1:
                    conflicts.append({"entry_id": entry_id, "class": cls,
                                      "slot": name, "values": ranked,
                                      "reason": "no unique Arnold stem"})
        # Unlike competing stem analyses, multiple forms in one imperative
        # PNG cell are licensed alternatives and must all remain generatable.
        for name in ("IPV_MS", "IPV_FS", "IPV_MP", "IPV_FP"):
            for stem in dict.fromkeys(values[(entry_id, cls)].get(name, ())):
                slots.append(StemSlot(name, stem, "EXPLICIT"))
        # A final =l before a following definite object is written phonetically
        # as əl in Arnold's running example (maylefəl ebre), but it is not part
        # of the IV present stem. Accept the correction only when removing əl
        # gives exactly the class-model m- + imperfect stem.
        class_prs = "m" + imp_stem if clean else ""
        prs_index = next((i for i, slot in enumerate(slots)
                          if slot.name == "PRS_MS"), None)
        prs_boundary_normalized = False
        if cls == "IV" and prs_index is not None \
                and slots[prs_index].stem == class_prs + "əl":
            slots[prs_index] = StemSlot("PRS_MS", class_prs, "DERIVED")
            prs_boundary_normalized = True
        # "Verified" is decided by Arnold-extracted PNG stems ONLY, before any
        # seeding, so the round-trip guarantee that tag implies is preserved.
        has_png_stem = any(s.stem for s in slots if s.name in (
            "PST_SUFFIXED", "SBJV_STEM", "SBJV_OBLIQUE",
            "PRS_OBLIQUE", "PRF_OBLIQUE"))
        # Seed the present participle for VII/VIII from the imperfect where
        # Arnold's inflected forms did not yield it (fṯḥ I7's present forms are
        # all progressive ʕam-, so nothing was extracted): the derived present
        # participle is m- + the imperfect stem (yinəfṯaḥ -> minəfṯaḥ). Restricted
        # to the passive/reflexive classes and well-formed stems.
        by_name = {s.name: i for i, s in enumerate(slots)}

        def _seed(name, value, status="DERIVED"):
            i = by_name.get(name)
            if i is not None and not slots[i].stem:
                slots[i] = StemSlot(name, value, status)
            elif i is None:
                by_name[name] = len(slots)
                slots.append(StemSlot(name, value, status))

        # Arnold II2/III2/IV2 principal parts explicitly include the
        # reflexive/passive č(i)- material (e.g. čṭawwar : yičṭawwar).
        # The imperfect minus its y- subject prefix is therefore the licensed
        # SBJV stem.  II2/IV2 PST person endings select the syncopated stem:
        # čṭawwar -> čṭawwr-aṯ; ordinary Maalula spelling may leave the
        # non-plosive ww gemination unwritten (čṭawr-aṯ).  III2 instead has
        # its own ablaut (čḥōrab -> čḥarib-innaḥ), completed below only for the
        # regular strong pattern.
        reflexive_principal_parts = cls in REFLEXIVE_DERIVED_CLASSES \
            and perfect.startswith("č") and imperfect.startswith("yič")
        if reflexive_principal_parts:
            _seed("PST_CITATION", perfect, "EXPLICIT")
            if cls != "III2":
                _seed("PST_SUFFIXED", _syncopate_theme(perfect))
            # imperfect is yi- + a cluster-initial reflexive stem (yičḥammaṣ);
            # the i after y is epenthetic before the cluster, so it belongs to
            # the person prefix (yi-), not the stem. Store the cluster-initial
            # stem (čḥammaṣ) so the generator's cluster->yi- rule yields the
            # canonical yi-čḥammaṣ, not y-ičḥammaṣ. III2 keeps the i-initial stem
            # for now: its extracted long-ī oblique and dedicated completion make
            # the cluster rewrite a separate, riskier reconciliation.
            _seed("SBJV_STEM",
                  imperfect[1:] if cls == "III2" else imperfect[2:], "EXPLICIT")

        if cls in ("I7", "I8") and clean:
            # the present participle is m- + the imperfect stem; the SBJV is left
            # alone (an n-initial stem's seeded 1sg n-inṣab is ambiguous).
            if perfect and "." not in perfect:
                _seed("PST_CITATION", perfect, "EXPLICIT")
            _seed("PRS_MS", "m" + imp_stem)
            _seed("PRS_OBLIQUE", "m" + imp_stem)
        # Paʕʕel (geminate C2): the PST-suffixed stem is the perfect with its
        # theme vowel syncopated (taxxal -> taxxl), which then surfaces with the
        # gemination unmarked (taxl-iṯ). Seed it where Arnold had none.
        if cls == "II" and perfect and "." not in perfect \
                and len(_consonants(perfect)) >= 2:
            _seed("PST_SUFFIXED", _syncopate_theme(perfect))
            # Strong II PC: y/č/n + qattel; before 2/3PL endings the theme
            # vowel syncopates (qattl-un/-an). Principal parts remain the
            # class-specific lexical carrier, so reverse analysis keeps the II
            # lexeme_id rather than borrowing a same-root G lexeme.
            root_cons = _consonants(source["root"] or "")
            strong_pc = len(root_cons) == 3 and not (set(root_cons) & WEAK_RADICALS)
            pc_stem = imp_stem or perfect
            if strong_pc and pc_stem:
                _seed("SBJV_STEM", pc_stem,
                      "EXPLICIT" if imp_stem else "DERIVED")
                _seed("SBJV_OBLIQUE", _syncopate_theme(pc_stem))
            # §3.4: the II present is m- + the class stem (mšattar), oblique
            # m- + the syncopated stem (mšattr-in). Seeds the čim-/nim- 2/1
            # present that the person prefix rides on.
            _seed("PRS_MS", "m" + perfect)
            _seed("PRS_OBLIQUE", "m" + _syncopate_theme(perfect))

        # A securely classed IV lexeme with an explicit PRS stem may still lack
        # object-free PST/SBJV examples in the dictionary. The two Arnold
        # principal parts explicitly supply those slots. Restrict this rescue
        # to the otherwise single-TAM profile, so established profiles do not
        # churn and a bare class label alone is insufficient.
        known_before_principal_parts = sum(
            any(s.name.startswith(t) and s.stem for s in slots)
            for t in ("PST", "SBJV", "PRS", "PRF", "IPV"))
        class_model_seeded = cls == "IV" and clean and prs_boundary_normalized \
            and known_before_principal_parts == 1 \
            and bool(next((s.stem for s in slots if s.name == "PRS_MS"), ""))
        if class_model_seeded:
            _seed("PST_CITATION", perfect, "EXPLICIT")
            _seed("SBJV_STEM", imp_stem, "EXPLICIT")

        # III (long-ō stem, §3.5): the two-stem ablaut is derived from the
        # citation. The unsuffixed stem keeps ō (sōfar), the pretonic suffixed
        # stem is C1aC2iC3 (safir, used before consonant-initial PST suffixes and
        # the SBJV plural), and the stressed suffixed stem is C1aC2īC3 (safīr,
        # used before the vowel-initial -aṯ — handled in the generator — and in
        # the PRS/PRF oblique). Extracted Arnold slots always win (_seed fills
        # only empties). III never geminates C2 (model qōtel/qatīla).
        if cls == "III" and perfect and "." not in perfect \
                and len(_consonants(perfect)) == 3:
            c1, c2, c3 = _consonants(perfect)
            pretonic = f"{c1}a{c2}i{c3}"
            stressed = f"{c1}a{c2}ī{c3}"

            def _force(name, value):
                i = by_name.get(name)
                if i is not None:
                    slots[i] = StemSlot(name, value, "DERIVED")
                else:
                    by_name[name] = len(slots)
                    slots.append(StemSlot(name, value, "DERIVED"))

            _seed("PST_CITATION", perfect, "EXPLICIT")
            _seed("SBJV_STEM", imp_stem or perfect, "EXPLICIT")
            _seed("PRS_MS", "m" + perfect)
            _seed("PRS_OBLIQUE", "m" + stressed)
            # the suffixed stems are fully determined by §3.5 (always a-i/ī),
            # so the grammar-conformant derivation is authoritative over the
            # transcription-noisy extraction (e.g. fōreḳ -> fariḳ, not fārik).
            _force("PST_SUFFIXED", pretonic)
            _force("SBJV_OBLIQUE", pretonic)

        # I10 (sč- stem, §2.12/§3.9): the default citation is sča-C1C2VC3; the
        # suffixed stem drops the theme vowel and epenthesises ə between C1 and
        # C2 (sčaʕmel -> sčaʕəml), the present adds the mi- marker (misčaʕmel).
        # The sč-/isč- variation is left to extraction; only the sča- default is
        # derived here.
        if cls == "I10" and perfect.startswith("sča") \
                and len(_consonants(perfect[3:])) == 3:
            c1, c2, c3 = _consonants(perfect[3:])
            suffixed = f"sča{c1}ə{c2}{c3}"
            _seed("PST_CITATION", perfect, "EXPLICIT")
            _seed("SBJV_STEM", imp_stem if imp_stem.startswith("sča") else perfect,
                  "EXPLICIT")
            _seed("PST_SUFFIXED", suffixed)
            _seed("SBJV_OBLIQUE", suffixed)
            _seed("PRS_MS", "mi" + perfect)
            _seed("PRS_OBLIQUE", "mi" + suffixed)

        # IV: the preterite and subjunctive share the a- stem (arceš : yarceš),
        # so a missing PST citation is the extracted SBJV stem. That stem also
        # carries the corpus transcription, so it is preferred for the
        # citation/lemma over the perfect's Arnold spelling when they differ
        # only by the k~c correspondence (roots_and_classes arkeš -> arceš).
        if cls == "IV":
            # Strong IV PC: the a-vocalism belongs to the person prefix in the
            # surface segmentation (ya-qtel / ča-qtel / na-qtel). The profile
            # keeps the traditional a-initial class stem so analyzer indexing
            # remains class-local; Generator moves the boundary at output.
            root_cons = _consonants(source["root"] or "")
            strong_pc = len(root_cons) == 3 and not (set(root_cons) & WEAK_RADICALS)
            pc_stem = imp_stem or perfect
            if strong_pc and pc_stem.startswith("a"):
                _seed("SBJV_STEM", pc_stem,
                      "EXPLICIT" if imp_stem else "DERIVED")
                _seed("SBJV_OBLIQUE",
                      "a" + _syncopate_theme(pc_stem[1:]))
            iv_sbjv = next((s.stem for s in slots
                            if s.name == "SBJV_STEM" and s.stem), "")
            if iv_sbjv:
                _seed("PST_CITATION", iv_sbjv, "DERIVED")
                if perfect and perfect != iv_sbjv \
                        and perfect.replace("c", "k") == iv_sbjv.replace("c", "k"):
                    perfect = iv_sbjv

        # IV2 (čč- passive, Arnold §2.10): the local table licenses the
        # complete finite core from the agreeing citation/imperfect pair.
        # Keep i/ə as stem allomorphs (never independent morphemes) and do not
        # invent an imperative, which Arnold explicitly leaves unavailable.
        iv2_complete = False
        if cls == "IV2" and perfect and imperfect:
            citation = _iv2_citation(perfect)
            sbjv = nfc(imperfect).translate(_ARNOLD_TRANSCRIPTION).replace("́", "")
            sbjv_stem = sbjv[1:] if sbjv.startswith("y") else sbjv
            sbjv_stem = sbjv_stem[1:] if sbjv_stem.startswith(("i", "í")) else sbjv_stem
            agrees = (citation.startswith("čč") and sbjv_stem.startswith("čč")
                      and _fold(_consonants(citation)) ==
                      _fold(_consonants(sbjv_stem)))
            if agrees:
                _seed("PST_CITATION", citation, "EXPLICIT")
                _seed("SBJV_STEM", sbjv_stem, "EXPLICIT")
                for value in _iv2_oblique_variants(citation):
                    if not any(s.name == "PST_SUFFIXED" and s.stem == value
                               for s in slots):
                        slots.append(StemSlot("PST_SUFFIXED", value, "DERIVED"))
                _seed("PRS_MS", "mi" + citation, "DERIVED")
                for value in _iv2_oblique_variants(citation):
                    value = "mi" + value
                    if not any(s.name == "PRS_OBLIQUE" and s.stem == value
                               for s in slots):
                        slots.append(StemSlot("PRS_OBLIQUE", value, "DERIVED"))
                prf_ms, prf_oblique = _iv2_prf_stems(citation)
                _seed("PRF_MS", prf_ms, "DERIVED")
                _seed("PRF_OBLIQUE", prf_oblique, "DERIVED")
                required = ("PST_CITATION", "PST_SUFFIXED", "SBJV_STEM",
                            "PRS_MS", "PRS_OBLIQUE", "PRF_MS", "PRF_OBLIQUE")
                iv2_complete = all(any(s.name == name and s.stem for s in slots)
                                   for name in required)

        known_tams = sum(any(s.name.startswith(t) and s.stem for s in slots)
                         for t in ("PST", "SBJV", "PRS", "PRF", "IPV"))
        if known_tams < 2:
            continue
        verification = (("VERIFIED_DERIVED_CLASS", "VERIFIED_IV2_COMPLETE")
                        if iv2_complete else
                        (() if cls == "IV2" else
                         (("VERIFIED_DERIVED_CLASS",) if has_png_stem else ())))
        pid = f"prof_{entry_id.removeprefix('dict_')}_{cls.lower()}"
        if reflexive_principal_parts:
            evidence = ("Arnold class label", "Arnold principal parts",
                        "Arnold II2/III2/IV2 class model") \
                       + (("IV2 finite paradigm; no IPV",)
                          if cls == "IV2" else ()) \
                       + (f"dictionary page {source['page']}",)
        elif class_model_seeded:
            evidence = ("Arnold class label", "Arnold principal parts",
                        "Arnold IV class model",
                        f"dictionary page {source['page']}")
        else:
            # Preserve the established provenance record for pre-existing
            # II/IV/I7/I8 profiles; the append-only artifact should not churn.
            evidence = ("Arnold class label", "Arnold explicit TAM/PNG forms",
                        f"dictionary page {source['page']}")
        profiles.append(LexemeMorphProfile(
            profile_id=pid, lexeme_id=f"{entry_id}:{cls}", lemma=perfect,
            pos=POS.VERB.value, pos_status="EXPLICIT",
            root=tuple(_consonants(source["root"] or "")),
            root_class=f"DERIVED_{cls}",
            derivational_stem=DERIVED_STEM_NAMES[cls],
            inflection_class=cls,
            stem_slots=tuple(slots),
            allowed_morphophonology=(
                (("UNWRITTEN_NONPLOSIVE_GEMINATION", "EPENTHESIS_I",
                  "EPENTHESIS_SCHWA") if cls == "IV2" else
                 ("UNWRITTEN_NONPLOSIVE_GEMINATION",))
                if reflexive_principal_parts else
                (("EPENTHESIS_I", "EPENTHESIS_SCHWA") if cls == "IV2" else ())),
            evidence=evidence + verification,
            conflicts=tuple(c["slot"] for c in conflicts
                            if c.get("entry_id") == entry_id and c.get("class") == cls),
        ))
    return profiles, conflicts


WEAK_RADICALS = set("wyʔ")
# Arnold's root field is transcribed with plain consonants (skl, tmr) while the
# inflected surfaces keep the emphatics (sḳul, ṭumr). Fold emphatic->plain for
# the root-vs-stem consonant identity check only; the emphatic-correct surface
# stem is always the one stored.
_EMPHATIC_FOLD = str.maketrans("ḳṣṭḏṯẓ", "kstdtz")


def _fold(cons: str) -> str:
    return cons.translate(_EMPHATIC_FOLD)


_VOWELS = "aeiouəáéíóúàèìòùāēīōū"


def _syncopate_theme(word: str) -> str:
    """Drop the theme vowel before the final consonant (taxxal -> taxxl,
    čṭawwar -> čṭawwr), the shape a PST person suffix attaches to."""
    return re.sub(rf"[{_VOWELS}](?=[^{_VOWELS}]$)", "", word)


def _complete_regular_strong_iii2(
        profile: LexemeMorphProfile) -> LexemeMorphProfile:
    """Complete the locally attested finite core of strong III2 verbs.

    Arnold's agreeing principal parts license ``čCōCa/eC : yičCōCa/eC``.
    His explicit inflected forms independently attest the class ablaut:

    * PST suffixed ``čCaCiC-`` (čḥarib-innaḥ);
    * SBJV singular/1PL ``ičCōCa/eC`` and plural ``ičCaCīC-``;
    * PRS ``mičCōCa/eC`` / ``mičCaCīC-``;
    * PRF oblique ``čCaCīC-``.

    General PNG exponents come from the local conjugation table and are
    attached by Generator.  PRF_MS is preserved only when explicitly known
    (e.g. čkōʕet), and IPV remains UNKNOWN: the sole extracted III2 imperative
    is weak and does not license a productive strong pattern.
    """
    if profile.inflection_class != "III2":
        return profile
    citation = profile.slot("PST_CITATION") or ""
    match = re.fullmatch(
        rf"č([{CONS}])ō([{CONS}])([ae])([{CONS}])", citation)
    if not match:
        return profile
    c1, c2, _theme, c3 = match.groups()
    root = "".join(profile.root or ())
    if len(root) != 3 or any(radical in WEAK_RADICALS for radical in root):
        return profile
    if _fold(root) != _fold(c1 + c2 + c3):
        return profile

    expected = {
        "PST_SUFFIXED": f"č{c1}a{c2}i{c3}",
        # III2 keeps the i-initial SBJV stems for now (consistent with the
        # extracted long-ī oblique); its yi- boundary rewrite is deferred.
        "SBJV_STEM": f"i{citation}",
        "SBJV_OBLIQUE": f"ič{c1}a{c2}ī{c3}",
        "PRS_MS": f"mi{citation}",
        "PRS_OBLIQUE": f"mič{c1}a{c2}ī{c3}",
        "PRF_OBLIQUE": f"č{c1}a{c2}ī{c3}",
    }
    existing = {slot.name: slot for slot in profile.stem_slots}
    # A directly extracted allomorph wins over the class model.  Such a lexeme
    # stays exact-only rather than acquiring a partly incompatible paradigm.
    for name, stem in expected.items():
        current = existing.get(name)
        if current and current.stem and current.stem != stem:
            return profile

    slots = list(profile.stem_slots)
    by_name = {slot.name: i for i, slot in enumerate(slots)}
    for name, stem in expected.items():
        i = by_name.get(name)
        if i is None:
            by_name[name] = len(slots)
            slots.append(StemSlot(name, stem, "DERIVED"))
        elif not slots[i].stem:
            slots[i] = StemSlot(name, stem, "DERIVED")

    markers = (
        "Arnold III2 strong principal-part and inflected-form model",
        "Glossing_updated.docx table 1 finite PNG exponents",
        "VERIFIED_III2_STRONG",
    )
    evidence = profile.evidence + tuple(
        marker for marker in markers if marker not in profile.evidence)
    return replace(profile, stem_slots=tuple(slots), evidence=evidence)


def _complete_regular_strong_g_imperative(
        profile: LexemeMorphProfile) -> LexemeMorphProfile:
    """Add the regular G imperative variants licensed by grammar column M.

    The local table ``imperative I stemm.png`` gives the Maalula (M) paradigm
    for the u-theme ``ḳṭl`` and a-theme ``ʕbr`` models:

    * u: iCCuC/CCōC (MS), iCCuC/CCūC (FS);
    * a: iCCaC/CCāC (common singular);
    * both: CuCCun/CuCCōn (MP), CuCCen/CuCCēn (FP).

    This is deliberately restricted to a strong triliteral whose exact
    ``SBJV_STEM`` has the clean CC[ua]C shape and agrees with the recorded
    root. Explicit per-lexeme forms are retained; the class model only adds
    licensed alternatives that are not already present.
    """
    if profile.root_class != "STRONG_G":
        return profile
    sbjv = profile.slot("SBJV_STEM") or ""
    match = re.fullmatch(
        rf"([{CONS}])([{CONS}])([ua])([{CONS}])", sbjv)
    if not match:
        return profile
    c1, c2, theme, c3 = match.groups()
    root = "".join(profile.root or ())
    if len(root) != 3 or any(radical in WEAK_RADICALS for radical in root):
        return profile
    if _fold(root) != _fold(c1 + c2 + c3):
        return profile

    short_singular = f"i{c1}{c2}{theme}{c3}"
    singular = {
        "IPV_MS": (
            short_singular,
            f"{c1}{c2}{'ō' if theme == 'u' else 'ā'}{c3}",
        ),
        "IPV_FS": (
            short_singular,
            f"{c1}{c2}{'ū' if theme == 'u' else 'ā'}{c3}",
        ),
    }
    shifted = f"{c1}u{c2}{c3}"
    derived = {
        **singular,
        "IPV_MP": (f"{shifted}un", f"{shifted}ōn"),
        "IPV_FP": (f"{shifted}en", f"{shifted}ēn"),
    }
    slots = list(profile.stem_slots)
    existing = {(slot.name, slot.stem) for slot in slots if slot.stem}
    for name, variants in derived.items():
        for stem in variants:
            if (name, stem) not in existing:
                slots.append(StemSlot(name, stem, "DERIVED"))
                existing.add((name, stem))

    marker = (
        "Arnold grammar imperative I stemm.png column M regular G model")
    evidence = profile.evidence
    if marker not in evidence:
        evidence += (marker,)
    return replace(profile, stem_slots=tuple(slots), evidence=evidence)


_PLOSIVES = set("bptṭdkḳgqʔčjc")


def _degeminate_nonplosive(s: str) -> str:
    out: list[str] = []
    for ch in s:
        if out and out[-1] == ch and ch not in _PLOSIVES:
            continue
        out.append(ch)
    return "".join(out)


def _established_class_i_ids(root: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """(dictionary entry, lemma) -> (profile id, lexeme id) of the class-I
    profiles the profile file already holds."""
    path = root / "data/grammar_kb/morphology_v4/lexeme_profiles.jsonl"
    out: dict[tuple[str, str], tuple[str, str]] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if '"prof_ci_' not in line:
            continue
        row = json.loads(line)
        lexeme_id = str(row.get("lexeme_id") or "")
        if str(row.get("profile_id") or "").startswith("prof_ci_") and lexeme_id:
            out.setdefault((lexeme_id.split(":")[0], nfc(row.get("lemma") or "")),
                           (row["profile_id"], lexeme_id))
    return out


def _class_i_strong_profiles(root: Path) -> tuple[list[LexemeMorphProfile], list[dict]]:
    """Basic G-stem (Arnold class I) strong triliteral verbs from the dictionary.

    This is the systemic gap behind unanalysable SBJV forms like yisḳul / yṭumrun:
    only ~40 strong-G verbs were reachable (train-attested), while Arnold labels
    hundreds explicitly as class I. Each such verb gets the whole SBJV
    person-prefix paradigm (y(i)-CCuC, -un/-an) plus the PST citation:

      SBJV_STEM   from the imperfect (yisḳul -> sḳul), else derived C1C2uC3
      PST_CITATION from the perfect (isḳal)

    A derived stem is never emitted blindly: the analyzer only keeps a generated
    form that reconstructs the input exactly, so a wrong theme vowel simply fails
    to match rather than producing a false analysis.
    """
    profiles, conflicts = [], []
    minted: set[tuple[str, str]] = set()
    ipv_by_entry: dict[str, list[StemSlot]] = defaultdict(list)
    for line in (root / "data/dictionary_morphology_v1/inflected_forms.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        form = json.loads(line)
        if form.get("arnold_stem_class") != "I" \
                or form.get("tam_canonical") != "IPV" \
                or not (form.get("selected") and form.get("usable_for_maalula")) \
                or form.get("effective_dialect") not in ("M", None) \
                or form.get("object_features") or form.get("proclitic"):
            continue
        features = form.get("features") or {}
        slot_name = {
            ("S", "M"): "IPV_MS", ("S", "F"): "IPV_FS",
            ("P", "M"): "IPV_MP", ("P", "F"): "IPV_FP",
        }.get((features.get("number"), features.get("gender")))
        surface = nfc(form.get("surface_canonical") or "").translate(
            _ARNOLD_TRANSCRIPTION)
        if not slot_name or not surface or any(mark in surface for mark in "-=‿ "):
            continue
        for variant in (part.strip() for part in surface.split("/")):
            if variant:
                ipv_by_entry[form["source_entry_id"]].append(
                    StemSlot(slot_name, variant, "EXPLICIT"))

    # Ids already given out stay with their verbs (the profile file is
    # append-only): a verb the dictionary parser finds later, on an earlier
    # page, must not take over the id of one that has it.
    established = _established_class_i_ids(root)
    taken = {profile_id for profile_id, _ in established.values()}
    taken_lexemes = {lexeme_id for _, lexeme_id in established.values()}
    for line in (root / "data/dictionary_morphology_v1/roots_and_classes.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        # a root given dialect by dialect (čbʕ/ćbʕ, ḥšb/p, žbl/ǧbl): Maalula's
        # form stands first -- read whole it had six consonants and the
        # verb (ičbaʕ, iḥšab, ižbal) no profile
        cons = _consonants((row.get("root") or "").split("/")[0])
        # strong triliteral only: exactly three radicals, none of them weak
        if len(cons) != 3 or set(cons) & WEAK_RADICALS or len(set(cons)) < 2:
            continue
        for stem in row.get("stems") or []:
            if stem.get("class") != "I":
                continue
            perfect = nfc(stem.get("perfect") or "").split("/")[0].strip()
            imperfect = nfc(stem.get("imperfect") or "").split("/")[0].strip()
            sbjv = ""
            if imperfect.startswith("yi"):
                sbjv = imperfect[2:]
            elif imperfect.startswith("y") and len(imperfect) >= 4:
                sbjv = imperfect[1:]
            # keep the emphatic-correct imperfect stem if its skeleton matches the
            # (plain-transcribed) root; otherwise derive a u-theme stem
            derived = not (sbjv and _fold(_consonants(sbjv)) == _fold(cons))
            if derived:
                sbjv = f"{cons[0]}{cons[1]}u{cons[2]}"   # u-theme default
            if _fold(_consonants(sbjv)) != _fold(cons):
                continue
            slots = [StemSlot("SBJV_STEM", sbjv, "DERIVED" if derived else "EXPLICIT")]
            if perfect and _fold(_consonants(perfect)) == _fold(cons) and "-" not in perfect:
                slots.append(StemSlot("PST_CITATION", perfect, "EXPLICIT"))
            slots.extend(ipv_by_entry.get(row["entry_id"], ()))
            lemma = perfect or sbjv
            known = established.get((row["entry_id"], lemma))
            if known:
                pid, lexeme_id = known
            else:
                # A verb with no profile yet. The id keeps only the letters
                # a-z, ʕ, ḥ, ṣ, ṭ, ḏ, ṯ, ẓ of the root, so š, ḳ, č, ž, ġ fall
                # out of it: ifraḳ, ifraš and ifraž all came to prof_ci_fr and
                # build_profiles() kept the first of an id -- 129 of the 547
                # strong class-I verbs of the dictionary had no profile
                # (ifraḳ, inšaḳ, ilḳaṭ, irḳaḏ, ilḥaḳ, iḳbar ...). A free id is
                # taken as it is, a taken one gets the entry added. Only a
                # perfect of the ground-stem shape is given a profile: among
                # the dropped rows are imperfects and notes read as perfects
                # (yišruḳ, "intr.").
                spelled = perfect.translate(_ARNOLD_TO_PROJECT)
                shape = _G_CITATION.match(spelled)
                if not shape or shape.group(1) == shape.group(2) \
                        or _fold(_consonants(spelled)) != _fold(cons):
                    continue
                pid = "prof_ci_" + re.sub(
                    r"[^a-zāēīōūʕḥṣṭḏṯẓ]", "",
                    (row.get("root_original") or cons).lower())[:22]
                if pid in taken:
                    unique = f"{pid}_{str(row['entry_id']).removeprefix('dict_')[:8]}"
                    pid, n = unique, 2
                    while pid in taken:
                        pid, n = f"{unique}_{n}", n + 1
                # a second ground stem of one entry (iḳṭaʕ yiḳṭuʕ tr. beside
                # iḳṭeʕ yiḳṭaʕ intr.) is a lexeme of its own
                lexeme_id, n = f"{row['entry_id']}:I", 2
                while lexeme_id in taken_lexemes:
                    lexeme_id, n = f"{row['entry_id']}:I:{n}", n + 1
                if (row["entry_id"], lemma) in minted:
                    continue                    # the same verb listed twice
                minted.add((row["entry_id"], lemma))
            taken.add(pid)
            taken_lexemes.add(lexeme_id)
            profiles.append(LexemeMorphProfile(
                profile_id=pid, lexeme_id=lexeme_id, lemma=lemma,
                pos=POS.VERB.value, pos_status="EXPLICIT",
                root=tuple(cons), root_class="STRONG_G",
                derivational_stem="G", inflection_class="STRONG_G_CLASS_I",
                stem_slots=tuple(slots),
                allowed_morphophonology=("EPENTHESIS_I", "PREFIX_YI_YA"),
                evidence=("Arnold class I label",
                          f"dictionary page {row.get('page')}",
                          "imperfect principal part" if not derived else "u-theme derived")))
    return profiles, conflicts


def _train_ids(root: Path) -> set:
    return {json.loads(l)["example_id"]
            for l in (root / "data/evaluation/split_examples.jsonl")
            .read_text().splitlines() if l.strip()
            and json.loads(l).get("split") == "train"}


def build_profiles(root: Path) -> tuple[list[LexemeMorphProfile], list[dict]]:
    train = _train_ids(root)
    verbs = _verb_profiles_from_cells(root)
    strong, sconf = _strong_g_profiles_from_train(root, train)
    arnold, aconf = _arnold_verb_slots(root)
    derived, dconf = _derived_verb_profiles(root)
    # Arnold sometimes lists one verb under two entry_ids, yielding twin profiles
    # (same lemma+class) with complementary slots and distinct lexeme_ids; the
    # analyzer can only return one, so the twin fails a per-lexeme_id round-trip.
    # Merge twins into a single lexeme_id, filling empty slots from the other.
    merged: dict[tuple, LexemeMorphProfile] = {}
    order: list[tuple] = []
    for p in derived:
        key = (p.lemma, p.root_class)
        if key not in merged:
            merged[key] = p
            order.append(key)
            continue
        kept = merged[key]
        names = [s.name for s in kept.stem_slots]
        slots = list(kept.stem_slots)
        for s in p.stem_slots:
            if not s.stem:
                continue
            if s.name in names:
                i = names.index(s.name)
                if not slots[i].stem:
                    slots[i] = s
            else:
                names.append(s.name)
                slots.append(s)
        merged[key] = replace(kept, stem_slots=tuple(slots))
    derived = [merged[k] for k in order]
    reflexive_derived = [
        _complete_regular_strong_iii2(p)
        for p in derived if p.inflection_class in REFLEXIVE_DERIVED_CLASSES]
    derived = [
        p for p in derived if p.inflection_class not in REFLEXIVE_DERIVED_CLASSES]
    class_i, ciconf = _class_i_strong_profiles(root)
    # merge Arnold PRS/PRF oblique stems into a train strong-G profile of the
    # same root, so the generator can produce that lexeme's other PNG forms
    for p in strong:
        rc = "".join(p.root or ())
        extra = arnold.get(rc, {}).get("slots") if rc else None
        if not extra:
            continue
        have = {s.name for s in p.stem_slots}
        new_slots = tuple(p.stem_slots) + tuple(
            StemSlot(n, s, "DERIVED") for n, s in extra.items() if n not in have)
        if len(new_slots) != len(p.stem_slots):
            strong[strong.index(p)] = replace(
                p, stem_slots=new_slots, evidence=p.evidence + ("Arnold",))
    strong = [_complete_regular_strong_g_imperative(p) for p in strong]
    class_i = [_complete_regular_strong_g_imperative(p) for p in class_i]
    nouns, conflicts = _noun_profiles_from_corpus(root)
    curated = [
        _complete_regular_strong_iii2(p) for p in _curated_profiles(root)]
    conflicts = conflicts + sconf + aconf + dconf + ciconf
    # train/derived profiles win a lemma collision (they are attested); class-I
    # Arnold profiles come last so they only add genuinely new lexemes.
    # New source classes are appended after the established profile sequence,
    # keeping the generated JSONL stable and append-only.
    profiles = (verbs + strong + derived + class_i + [_tarba_profile()] + nouns
                + reflexive_derived + curated)
    # dedup by profile_id, keep first
    seen, unique = set(), []
    for p in profiles:
        if p.profile_id in seen:
            continue
        seen.add(p.profile_id)
        unique.append(p)
    return unique, conflicts


# Arnold's transcription -> the project's (config/transcription_map.yaml):
# ḳ -> k and k -> c, simultaneously.
_ARNOLD_TO_PROJECT = str.maketrans({"ḳ": "k", "k": "c"})


def _project_spelling(profile: LexemeMorphProfile) -> LexemeMorphProfile:
    """Spell stems still written in Arnold's system the way the texts are.

    Some builder paths copy a stem from Arnold's dictionary without the
    ḳ -> k, k -> c mapping (199 profiles in 2026-09 data: šaḳl-type stems,
    ʕappeḳ, ʕaḳḳam, ʕōḳeb...), so a form generated from them can only match a
    word typed with ḳ, which no text does: ``mʕappek`` came out
    ``m=ʕappek from=ʕappeḳ`` while ``mʕappeḳ`` was the II-stem NC. A stem
    that contains ḳ is unambiguously in Arnold's system, so only those stems
    (and the root, when a stem was) are respelled; a stem without ḳ is left
    alone, since its k may already be the project's k.
    """
    if not any("ḳ" in (slot.stem or "") for slot in profile.stem_slots):
        return profile
    slots = tuple(replace(slot, stem=nfc(slot.stem).translate(_ARNOLD_TO_PROJECT))
                  if "ḳ" in (slot.stem or "") else slot for slot in profile.stem_slots)
    root = profile.root
    if root and "ḳ" in "".join(root):
        root = tuple(nfc(c).translate(_ARNOLD_TO_PROJECT) for c in root)
    return replace(profile, stem_slots=slots, root=root)


_G_CITATION = re.compile(
    r"^i?([^aeiouāēīōūə\u0323]\u0323?)([^aeiouāēīōūə\u0323]\u0323?)([ae])"
    r"([^aeiouāēīōūə\u0323]\u0323?)$")


def _g_suffixed_base(profile: LexemeMorphProfile) -> LexemeMorphProfile:
    """Seed the suffixed preterite base of a strong G verb from its citation.

    Arnold § 4.1.1.1 (S. 70): the 3 sg m and 3 pl c take no suffix on the
    base KKvK- (with the prothetic i-: ifṯaḥ, iḏmex); every other preterite
    form is built on KvKK-: faṯḥ-aṯ, faṯḥ-ič, faṯḥ-iṯ, faṯḥ-innaḥ with base
    vowel a, ḏimx-aṯ, ḏimx-iṯ with e/i (S. 71). 444 of the 456 strong-G
    profiles had no such slot, so the engine could not build faṯḥaṯ, šaklaṯ,
    kaṭʕaṯ as G at all, and the only reading it found was a II verb with its
    gemination unwritten (faṯṯaḥ, šakkel, ḳaṭṭaʕ). The corpus-derived bases
    agree with the rule (ṯikn, silk, šamṭ, niḥč from iṯken, islek, išmaṭ,
    inḥeč). The citation is PST_CITATION, else the lemma of a dictionary
    profile; a doubled first cluster (iffek < nfk) is left alone. A base the
    profile already has is never replaced.

    The lemma of a dictionary profile is still in Arnold's transcription
    (the builder stores no PST_CITATION when the perfect has Arnold's k and
    the root the project's c), so it is respelled first: Arnold's iškal
    "sticken, annähen" is išcal with the base šacl-. Read as it stands it
    gave šakl-, and šaklaṯ "she took" (išḳal) was credited to this verb.
    """
    if profile.root_class != "STRONG_G" or profile.slot("PST_SUFFIXED"):
        return profile
    citation = nfc(profile.slot("PST_CITATION") or "")
    if not citation and profile.inflection_class == "STRONG_G_CLASS_I":
        citation = nfc(profile.lemma or "").translate(_ARNOLD_TO_PROJECT)
        if _fold(_consonants(citation)) != _fold("".join(profile.root or ())):
            return profile
    match = _G_CITATION.match(citation)
    if not match or match.group(1) == match.group(2):
        return profile
    first, second, vowel, third = match.groups()
    base = first + ("a" if vowel == "a" else "i") + second + third
    slot = StemSlot(name="PST_SUFFIXED", stem=base, status="DERIVED")
    return replace(profile, stem_slots=tuple(profile.stem_slots) + (slot,),
                   evidence=tuple(profile.evidence)
                   + ("PST_SUFFIXED from citation, Arnold § 4.1.1.1",))


class ProfileStore:
    def __init__(self, profiles: list[LexemeMorphProfile]):
        self.profiles = profiles
        self.by_lemma: dict[str, LexemeMorphProfile] = {}
        self.by_id: dict[str, LexemeMorphProfile] = {}
        self.by_lexeme_id: dict[str, LexemeMorphProfile] = {}
        self.by_stem: dict[str, list[LexemeMorphProfile]] = defaultdict(list)
        from .object_transducer import _syncopate
        from .morphophonology import simplify_geminate_before_consonant
        for p in profiles:
            self.by_id[p.profile_id] = p
            if p.lexeme_id:
                self.by_lexeme_id[p.lexeme_id] = p
            self.by_lemma.setdefault(nfc(p.lemma), p)
            for slot in p.stem_slots:
                if slot.stem:
                    self.by_stem[nfc(slot.stem)].append(p)
                    # the II/IV subjunctive drops its base vowel before a
                    # plural ending (Arnold § 4.1.2.2, § 4.1.4.2: yšattrun,
                    # yaḥrfun), and most profiles store no such stem:
                    # shortlist the profile under the syncopated base too
                    stems = [nfc(slot.stem)]
                    if slot.name == "SBJV_STEM" and p.inflection_class in ("II", "IV"):
                        short = _syncopate(nfc(slot.stem))
                        if short:
                            stems.append(short)
                    # also index the stem as the texts often write it: a
                    # non-plosive gemination unmarked (taxl-), any geminate
                    # before a consonant single (šattr- -> šatr-, mšakkl- ->
                    # mšakl-), so such a surface still shortlists the profile
                    for stem in list(stems):
                        stems += [_degeminate_nonplosive(stem),
                                  simplify_geminate_before_consonant(stem)]
                    for stem in dict.fromkeys(stems):
                        if p not in self.by_stem[stem]:
                            self.by_stem[stem].append(p)

    def resolve(self, key: str) -> LexemeMorphProfile | None:
        return (self.by_id.get(key) or self.by_lexeme_id.get(key)
                or self.by_lemma.get(nfc(key)))

    @classmethod
    def load(cls, root: Path) -> "ProfileStore":
        path = root / "data/grammar_kb/morphology_v4/lexeme_profiles.jsonl"
        profiles = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            profiles.append(LexemeMorphProfile(
                profile_id=d["profile_id"], lemma=d["lemma"], pos=d["pos"],
                lexeme_id=d.get("lexeme_id"),
                pos_status=d.get("pos_status", "INFERRED"),
                gender=d.get("gender"),
                root=tuple(d["root"]) if d.get("root") else None,
                root_class=d.get("root_class"),
                derivational_stem=d.get("derivational_stem"),
                inflection_class=d.get("inflection_class"),
                paradigm_cell_ids=tuple(d.get("paradigm_cell_ids") or ()),
                allowed_morphophonology=tuple(d.get("allowed_morphophonology") or ()),
                stem_slots=tuple(StemSlot(**s) for s in d.get("stem_slots") or ()),
                evidence=tuple(d.get("evidence") or ())))
        profiles = [_g_suffixed_base(_project_spelling(profile)) for profile in profiles]
        # Curated slots are runtime data as well as rebuild input. Keep all
        # generated slots not explicitly overridden by the curated profile.
        by_id = {profile.profile_id: index for index, profile in enumerate(profiles)}
        for curated in _curated_profiles(root):
            if curated.profile_id in by_id:
                index = by_id[curated.profile_id]
                previous = profiles[index]
                names = {slot.name for slot in curated.stem_slots}
                # ... and the root, when the curated row states one: a profile
                # built from the texts reads it off the citation, and the
                # citation of iffek has lost the n of nfk
                # A curated slot with an empty stem takes the generated slot
                # of that name away and puts nothing in its place: the profiles
                # built from the texts filed the present (sōlek) under PRF_MS.
                merged = curated.stem_slots + tuple(
                    slot for slot in previous.stem_slots if slot.name not in names)
                profiles[index] = replace(
                    previous, stem_slots=tuple(slot for slot in merged if slot.stem),
                    root=curated.root or previous.root)
            else:
                by_id[curated.profile_id] = len(profiles)
                profiles.append(curated)
        return cls(profiles)
