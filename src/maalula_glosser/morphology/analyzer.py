"""Reverse analyzer: surface -> analysis by exact lookup and constrained
generation (never a free root×pattern search).

MorphologyEngine ties the profile store, the exact paradigm store and the
generator together and is the self-contained v4 entry point used in v4/shadow
mode. It returns only candidates whose generated surface reconstructs the input.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from maalula_glosser.normalization.gloss import render_authoritative_gloss
from maalula_glosser.normalization.transcription import fold_allophones

from .generator import Generator, GeneratedForm
from .morphophonology import (respell_devoiced_b,
                              simplify_geminate_before_consonant,
                              voice_p_after_consonant)
from .lexeme_profiles import ProfileStore
from .object_transducer import (DERIVED_OBJECT_KINDS, ObjectTransducer, SOURCE_CONTROLS,
                                object_key, written_readings)
from .paradigm_store import ParadigmStore


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value or "")


@dataclass
class V4Candidate:
    surface: str
    segmentation: str
    gloss: str
    lexeme: str
    lexeme_id: str | None
    features: dict
    evidence: str
    reconstruction_ok: bool
    confidence: str = "MEDIUM"
    morphophonology: list[dict] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    # the text does not spell the generated form letter for letter (a
    # geminate unwritten, a linker restored): ranked after what it does spell
    respelt: bool = False

    def as_dict(self) -> dict:
        d = {"surface": self.surface, "segmentation": self.segmentation,
                "gloss": self.gloss, "lexeme": self.lexeme,
                "lexeme_id": self.lexeme_id,
                "features": self.features, "evidence": self.evidence,
                "reconstruction_ok": self.reconstruction_ok,
                "confidence": self.confidence,
                "morphophonology": self.morphophonology}
        d["trace"] = self.trace
        return d


def _bare(surface: str) -> str:
    return nfc(surface).replace("-", "").replace("=", "")


# Maalula writing frequently leaves the gemination of a NON-plosive consonant
# unmarked (taxxal -> taxl, čṭawwar -> čṭawr), while plosive gemination is kept.
# A generated geminate form may therefore reconstruct a single-consonant surface.
_PLOSIVES = set("bptṭdkḳgqʔčjc")


_SHORTEN = {"ā": "a", "ē": "e", "ī": "i", "ō": "o", "ū": "u"}


def _shorten_final_long(stem: str) -> str:
    """Shorten the last long vowel (safīr -> safir); inverse of the III stress
    lengthening used in generation."""
    for i in range(len(stem) - 1, -1, -1):
        if stem[i] in _SHORTEN:
            return stem[:i] + _SHORTEN[stem[i]] + stem[i + 1:]
    return stem


def _degeminate(s: str) -> str:
    out: list[str] = []
    for ch in s:
        if out and out[-1] == ch and ch not in _PLOSIVES:
            continue
        out.append(ch)
    return "".join(out)


def _yields_to_the_present(candidate: V4Candidate) -> bool:
    """A preterite reading of -il=l- / -al=l that the present shares.

    kasm-il=le is "they divide it" (kōsmin + le) far more often than "I
    divided for him" (kasmiṯ + le): glossed_all has a participle 54 times in
    -il=l-, the 1 sg preterite 8, all of amar and aḥəc "say, tell". The 3 pl f
    preterite before the DOM (xalṭ-al=l) is rarer still than the present.
    """
    features = candidate.features or {}
    if features.get("tam") != "PST":
        return False
    if features.get("person") == 1 and features.get("number") == "SG":
        return features.get("object_role") == "INDIRECT"
    return (features.get("person") == 3 and features.get("number") == "PL"
            and features.get("gender") == "F"
            and features.get("object_role") == "DOM")


def _is_preterite_with_clitic_a(candidate: V4Candidate) -> bool:
    """KaKK=a: the 3 sg m preterite with the bare 3 sg f suffix."""
    features = candidate.features or {}
    return (features.get("tam") == "PST" and features.get("person") == 3
            and features.get("number") == "SG" and features.get("gender") == "M"
            and features.get("object_role") == "DIRECT"
            and not features.get("linker")
            and candidate.segmentation.endswith("=a"))


def _is_first_singular_with_second_plural(candidate: V4Candidate) -> bool:
    """KaKK-ič=xun "I ... you": the same letters as the plain 2 pl KaKK-ičxun."""
    features = candidate.features or {}
    obj = features.get("object_features") or {}
    return (features.get("tam") == "PST" and features.get("person") == 1
            and features.get("number") == "SG"
            and features.get("object_role") == "DIRECT"
            and isinstance(obj, dict) and obj.get("person") == 2
            and obj.get("number") == "PL")


def _ranked(cells: list[V4Candidate], objects: list[V4Candidate],
            productive: list[V4Candidate],
            respelt: list[V4Candidate]) -> list[V4Candidate]:
    """One order for everything `analyze` found.

    Exact paradigm cells; then what spells the word letter for letter, the
    object forms first; then what needs a respelling (a geminate the text
    does not write, a linker it writes single: wark-an lira-FP before
    wark-ann "he ... them", tarb-a road-FREE before tarb=a of tarreb).

    Two object readings give way to a productive one that spells the word
    as well:

    * KaKK=a "he ...ed her" to a noun or adjective KaKK-a (fark-a
      difference-FREE): a word of this shape that is a known noun is the
      noun far more often;
    * KaKK-ič=xun "I ... you" to the plain 2 pl KaKK-ičxun.

    Nothing is dropped: the reading that gives way stays an alternative.
    """
    nominal = any(c.evidence.startswith(("PRODUCTIVE_NOUN", "PRODUCTIVE_ADJ"))
                  for c in productive)
    second_plural = any(
        (c.features or {}).get("tam") == "PST" and (c.features or {}).get("person") == 2
        and (c.features or {}).get("number") == "PL" for c in productive)

    def gives_way(candidate: V4Candidate) -> bool:
        return ((nominal and _is_preterite_with_clitic_a(candidate))
                or (second_plural and _is_first_singular_with_second_plural(candidate)))

    spelled = [c for c in objects if not c.respelt]
    return (cells
            + [c for c in spelled if not gives_way(c)]
            + productive
            + [c for c in spelled if gives_way(c)]
            + [c for c in objects if c.respelt]
            + respelt)


class MorphologyEngine:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.store = ParadigmStore(
            self.root / "data/grammar_kb/morphology_v4/paradigm_cells.jsonl")
        self.profiles = ProfileStore.load(self.root)
        self.objects = ObjectTransducer(self.root)
        self.gen = Generator(self.profiles, self.store, self.objects)
        self._object_form_index: dict[str, list[GeneratedForm]] | None = None
        self._derived_profiles: list | None = None

    # generate() delegates to the Generator (bidirectional core)
    def generate(self, lemma: str, features: dict) -> list[GeneratedForm]:
        return self.gen.generate(lemma, features)

    def analyze(self, surface: str) -> list[V4Candidate]:
        surface = fold_allophones(nfc(surface))
        out: list[V4Candidate] = []          # exact cells, closed items
        productive: list[V4Candidate] = []   # generated, spelled as written
        seen: set[tuple] = set()

        # 1. exact paradigm cell (irregular verb / object / normative)
        for cell in self.store.lookup_surface(surface):
            for seg in cell["surface_variants"]:
                seg = seg.replace("(", "").replace(")", "")
                key = (seg, cell["gloss_original"])
                if key in seen:
                    continue
                seen.add(key)
                out.append(V4Candidate(
                    surface=surface, segmentation=seg,
                    gloss=render_authoritative_gloss(cell["gloss_original"]),
                    lexeme=cell.get("lexeme_heading"),
                    lexeme_id=None,
                    features=cell["features"], evidence="EXACT_PARADIGM_CELL",
                    reconstruction_ok=_bare(seg) == _bare(surface),
                    confidence="MEDIUM",
                    trace=["lookup paradigm_cells.jsonl", "exact surface cell",
                           "authoritative gloss/features"]))

        # The independent object particle is a closed lexical item, not a
        # pronominal suffix on an invented verbal host.
        if _bare(surface) == "lēlun":
            segmentation, gloss = SOURCE_CONTROLS["lēlun"]
            key = (segmentation, gloss)
            if key not in seen:
                seen.add(key)
                out.append(V4Candidate(
                    surface=surface, segmentation=segmentation, gloss=gloss,
                    lexeme="lē", lexeme_id=None,
                    features={
                        "pos": "PARTICLE",
                        "object_role": "DO",
                        "object_features": {
                            "person": 3, "gender": "M", "number": "PL"},
                    },
                    evidence="EXACT_OBJECT_PARTICLE",
                    reconstruction_ok=True, confidence="HIGH",
                    trace=["object particle control", "source_controls=lēlun",
                           "exact reconstruction"]))

        # Productive strong-G object forms are generated from licensed profile
        # stems once, then indexed by their complete reconstructed surface.
        # This avoids both a brute-force root×pattern search and lossy suffix
        # stripping in reverse analysis.
        objects = self.object_readings(surface, seen)

        # 2. constrained productive generation from candidate profiles.
        # A form reached only by respelling an unwritten geminate comes after
        # every form spelled exactly as written: mrapʕa is mrapʕ-a room-FREE
        # (glossed_all) before it is m-rappʕ-a, the participle of rappaʕ.
        respelt_forms: list[V4Candidate] = []
        for profile in self._candidate_profiles(surface):
            for features in self._feature_hypotheses(profile, surface):
                for form in self.gen.generate(profile.profile_id, features):
                    seg = form.segmentation
                    respelt_spelling = _bare(form.surface) != _bare(surface)
                    if respelt_spelling:
                        # accept an unwritten gemination: a non-plosive one
                        # anywhere (taxxl-iṯ -> taxl-iṯ), any geminate before a
                        # consonant (y-šattr-un -> y-šatr-un, šakkl-aṯ ->
                        # šakl-aṯ: CxCx + C -> Cx C); the segmentation is
                        # respelled so it spells the surface as written.
                        # Likewise the b of the root written p after a
                        # consonant (ṭalb-aṯ -> ṭalp-aṯ): `_candidate_profiles`
                        # shortlists the profile under the b, and without this
                        # the form it generated was dropped here again.
                        respelt = next((spelled for spelled in (
                            respell_devoiced_b(candidate, surface)
                            for candidate in (
                                seg,
                                _degeminate(seg),
                                simplify_geminate_before_consonant(seg),
                                simplify_geminate_before_consonant(_degeminate(seg))))
                            if spelled), None)
                        if respelt is None:
                            continue
                        seg = respelt
                    # Preserve genuine syncretic alternatives (notably III2
                    # SBJV 1SG=1PL and PST 3MS=3PL).  Segmentation+gloss alone
                    # used to collapse them into one arbitrarily chosen PNG.
                    feature_key = tuple(sorted(
                        (name, repr(value))
                        for name, value in form.features.items()))
                    key = (seg, form.gloss, feature_key)
                    if key in seen:
                        continue
                    seen.add(key)
                    # A corpus-DERIVED noun/adj profile is keyed by the
                    # corpus's lexical gloss (see lexeme_profiles.py:130-135),
                    # so form.lexeme there is a meaning
                    # word ("car"), never a form. An EXPLICIT profile (the
                    # curated tarba noun) carries a real lemma and is exempt.
                    is_gloss_lexeme = (profile.pos_status == "DERIVED"
                                       and form.evidence in ("PRODUCTIVE_ADJ",
                                                              "PRODUCTIVE_NOUN"))
                    (respelt_forms if respelt_spelling else productive).append(V4Candidate(
                        surface=surface, segmentation=seg,
                        gloss=form.gloss,
                        lexeme=None if is_gloss_lexeme else form.lexeme,
                        lexeme_id=form.lexeme_id,
                        features=form.features, evidence=form.evidence,
                        reconstruction_ok=True,
                        morphophonology=form.morphophonology,
                trace=["lookup lexeme_profiles.jsonl", profile.profile_id,
                               "generator=" + form.evidence,
                               "surface reconstruction OK"]
                        + (["written otherwise (geminate single, b as p "
                            "after a consonant): " + form.surface]
                           if respelt_spelling else [])
                        + [str(step) for step in form.morphophonology],
                        respelt=respelt_spelling))
        return _ranked(out, objects, productive, respelt_forms)

    def object_readings(self, surface: str,
                        seen: set[tuple] | None = None) -> list[V4Candidate]:
        """Readings of ``surface`` as a verb with an object suffix or the DOM.

        The index is spelling-tolerant (object_key); each reading must still
        spell this very surface, stem as written, linker as the author cuts
        it (object_transducer.written_readings).
        """
        surface = fold_allophones(nfc(surface))
        seen = set() if seen is None else seen
        written = _bare(surface).replace("ə", "")
        out: list[V4Candidate] = []
        for form in self._object_forms(surface):
            feature_key = tuple(sorted(
                (name, repr(value)) for name, value in form.features.items()))
            stem = next((nfc(m.get("surface") or "") for m in form.overt_morphemes
                         if m.get("role") == "STEM"), "").replace("ə", "")
            for segmentation, gloss in written_readings(form, surface):
                key = (segmentation, gloss, feature_key)
                if key in seen:
                    continue
                seen.add(key)
                spelled = segmentation.replace("ə", "")
                as_written = written in (_bare(spelled),
                                         _bare(form.surface).replace("ə", ""))
                out.append(V4Candidate(
                    surface=surface, segmentation=segmentation,
                    gloss=gloss, lexeme=form.lexeme,
                    lexeme_id=form.lexeme_id, features=form.features,
                    evidence=form.evidence, reconstruction_ok=True,
                    morphophonology=form.morphophonology,
                    trace=["object transducer profile grid", form.evidence,
                           "exact generated surface"]
                    + [str(step) for step in form.morphophonology],
                    # a letter restored (šw-ann for šwan) or a stem geminate
                    # the text does not write (šaʕl- for šaʕʕl-)
                    respelt=not as_written or stem not in spelled))
        out.sort(key=lambda c: (c.respelt, _yields_to_the_present(c)))
        return out

    def _object_forms(self, surface: str) -> list[GeneratedForm]:
        if self._object_form_index is None:
            index: dict[str, list[GeneratedForm]] = {}
            for profile in self.profiles.profiles:
                if profile.root_class != "STRONG_G" and not any(
                        profile.slot(name + "_STEM") for name in ("NC", "NC1", "NC2")):
                    continue
                forms = self.gen.generate(
                    profile.profile_id, {"object_grid": True})
                for form in forms:
                    index.setdefault(object_key(form.surface), []).append(form)
            self._object_form_index = index
        key = object_key(surface)
        return self._object_form_index.get(key, []) + self._derived_object_forms(key)

    def _derived_object_forms(self, key: str) -> list[GeneratedForm]:
        """Object forms of the derived stems (II, III, IV), on demand.

        Their grid is several times larger than the strong-G one (more
        profiles, two bases per tense), so it is not pre-indexed: only the
        profiles whose root consonants occur in order in the word are
        generated, once each, and cached.
        """
        if self._derived_profiles is None:
            self._derived_profiles = [
                p for p in self.profiles.profiles
                if p.root_class in DERIVED_OBJECT_KINDS and p.root]
            self._derived_profiles += self._arnold_only_derived_profiles()
            self._derived_cache: dict[str, dict[str, list[GeneratedForm]]] = {}
        # the key reads p after a consonant as b (object_key); a root that
        # has a p of its own (blpš) must still find itself in it
        skeleton = [c.replace("p", "b") for c in key if c not in "aeiouāēīōūə"]
        out: list[GeneratedForm] = []
        for profile in self._derived_profiles:
            position = 0
            for radical in profile.root:
                try:
                    position = skeleton.index(radical.replace("p", "b"), position) + 1
                except ValueError:
                    break
            else:
                cached = self._derived_cache.get(profile.profile_id)
                if cached is None:
                    cached = {}
                    if self.profiles.resolve(profile.profile_id) is profile:
                        forms = self.gen.generate(profile.profile_id,
                                                  {"object_grid": True})
                    else:            # an Arnold-only verb, not in the store
                        forms = [GeneratedForm(
                            lexeme=profile.lemma, lexeme_id=profile.lexeme_id,
                            features=dict(form.features), surface=form.surface,
                            segmentation=form.segmentation, gloss=form.gloss,
                            overt_morphemes=list(form.overt_morphemes),
                            morphophonology=list(form.morphophonology),
                            evidence=f"PRODUCTIVE_{profile.derivational_stem}_OBJECT_ARNOLD",
                            reconstruction_ok=True)
                            for form in self.objects.generate_profile(
                                profile, {"object_grid": True})]
                    for form in forms:
                        cached.setdefault(object_key(form.surface), []).append(form)
                    self._derived_cache[profile.profile_id] = cached
                out.extend(cached.get(key, []))
        return out

    def _arnold_only_derived_profiles(self) -> list:
        """II / IV / III / quadriliteral verbs of Arnold's lexicon with no
        v4 profile (šaġġel, ʕarref, akreṭ, bahtel in Muše Sanžar): a
        minimal profile from the principal parts, used for the object grid
        only (`_derived_object_forms`), so nothing else in v4 changes.
        Quadriliterals follow the II stem (CaCCeC -> CaCCC before a vowel).
        """
        import re as _re
        from maalula_glosser.arnold_verbs import arnold_verbs
        from .lexeme_profiles import LexemeMorphProfile
        from .types import StemSlot
        classes = {"II": "DERIVED_II", "quad": "DERIVED_II",
                   "IV": "DERIVED_IV", "III": "DERIVED_III"}
        known = {nfc(p.lemma) for p in self.profiles.profiles}
        out = []
        for root, entries in arnold_verbs(self.root).items():
            radicals = tuple(_re.findall(r"[^\W\d_][\u0300-\u036f]*", nfc(root)))
            for entry in entries:
                root_class = classes.get(entry.get("stem_class"))
                words = nfc(entry.get("citation") or "").split()
                if not root_class or not words or len(radicals) < 3:
                    continue
                perfect = words[0]
                imperfect = words[1] if len(words) > 1 else ""
                if perfect in known or not _re.fullmatch(r"[^\W\d_]+", perfect):
                    continue
                sbjv = imperfect[1:] if imperfect.startswith("y") else perfect
                slots = [StemSlot("PST_CITATION", perfect, "DERIVED"),
                         StemSlot("SBJV_STEM", sbjv, "DERIVED"),
                         StemSlot("PRS_MS", "m" + sbjv, "DERIVED")]
                out.append(LexemeMorphProfile(
                    profile_id=f"arnold_{root_class.lower()}_{perfect}",
                    lemma=perfect, pos="VERB", pos_status="DERIVED",
                    root=radicals, root_class=root_class,
                    derivational_stem=entry.get("stem_class"),
                    stem_slots=tuple(slots),
                    evidence=(f"Arnold lexicon p.{entry.get('page')}",)))
                known.add(perfect)
        return out

    SBJV_PREFIXES = ("yi", "či", "ni", "y", "č", "n")

    def _candidate_profiles(self, surface: str):
        """Profiles whose stem could underlie this surface — no brute force.

        The word is tried as written and with p after a consonant read as b
        (y-ṭulp-un under ṭlub, ṭalp-aṯ under ṭalb); `analyze` keeps a form
        found the second way only if it spells the word with that one
        difference, and ranks it after what spells it exactly.
        """
        written = _bare(surface)
        seen, out = set(), []
        for bare in dict.fromkeys((written, voice_p_after_consonant(written))):
            self._shortlist(bare, seen, out)
        return out

    def _shortlist(self, bare: str, seen: set, out: list) -> None:

        def add(stem):
            for p in self.profiles.by_stem.get(stem, []):
                if p.profile_id not in seen:
                    seen.add(p.profile_id)
                    out.append(p)

        # a licensed nominal/adjectival/verbal ending stripped from the end
        for ending in ("yōṯa", "ōyṯa", "ōyin", "ōyan", "ōy", "ay",
                       "a", "ō", "in", "an", "e", "un", "ax", "iš", "xun", "hun",
                       "l", "il", "aṯ", "iṯ", "ič", "ičxun", "ičxen", "innaḥ"):
            if bare.endswith(ending) and len(bare) - len(ending) >= 2:
                stem = bare[: -len(ending)]
                add(stem)
                if stem.endswith("p"):
                    add(stem[:-1] + "b")          # devoiced tarp -> tarb
                # reverse the III stress lengthening so a long-ī suffixed stem
                # (safīr before -aṯ) is found under the stored pretonic stem safir
                short = _shorten_final_long(stem)
                if short != stem:
                    add(short)
                # reverse the m- participle tail metathesis so mižčamʕ(-in) is
                # found under the stored oblique stem mižčmaʕ
                if ending in ("in", "an", "a") and len(stem) >= 4 \
                        and stem[-3] in "uiəa":
                    add(stem[:-4] + stem[-4] + stem[-2] + stem[-3] + stem[-1])
        # a strong-G verb: peel a SBJV person prefix, then an optional -un/-an
        for prefix in self.SBJV_PREFIXES:
            if bare.startswith(prefix) and len(bare) - len(prefix) >= 3:
                rest = bare[len(prefix):]
                add(rest)
                for suf in ("a", "in", "un", "an"):
                    if rest.endswith(suf) and len(rest) - len(suf) >= 3:
                        stem = rest[: -len(suf)]
                        add(stem)
                        # reverse the CuCC -> CCuC resyllabification so ṭumr-un
                        # is looked up under the stored CCuC stem ṭmur
                        if len(stem) == 4 and stem[1] in "uiəa":
                            add(stem[0] + stem[2] + stem[1] + stem[3])
                        # reverse the longer m-participle tail shift after a
                        # person prefix (n-mižčamʕ-in -> mižčmaʕ)
                        if suf in ("a", "in", "an") and len(stem) >= 4 \
                                and stem[-3] in "uiəa":
                            add(stem[:-4] + stem[-4] + stem[-2]
                                + stem[-3] + stem[-1])
        if bare.startswith("m") and len(bare) >= 4:
            add(bare)
            add(bare[1:])
        add(bare)                                 # whole word (PST citation)

    @staticmethod
    def _feature_hypotheses(profile, surface: str):
        bare = _bare(surface)
        hyps: list[dict] = []
        if profile.pos == "NOUN":
            hyps += [{"number": "SG", "state": "FREE"},
                     {"number": "PL", "state": "FREE"},
                     {"number": "SG", "state": "CONSTRUCT"}]
            for poss in ("1S", "2MS", "2FS", "3MS", "3FS", "3MP", "2MP", "3P"):
                hyps.append({"possessor": poss})
        elif profile.pos == "ADJ":
            if profile.inflection_class == "ADJ_NISBA":
                for number in ("SG", "PL"):
                    for state in ("INDEFINITE", "DEFINITE"):
                        for gender in ("M", "F"):
                            hyps.append({"number": number, "state": state,
                                         "gender": gender})
            else:
                hyps += [{"number": "PL", "state": "INDEFINITE", "gender": "M"},
                         {"number": "PL", "state": "INDEFINITE", "gender": "F"},
                         {"number": "PL", "state": "DEFINITE", "gender": "M"}]
        elif profile.pos == "VERB" and any(
                profile.slot(name + "_STEM") for name in ("NC", "NC1", "NC2")):
            hyps += [{"tam": "PRS"}, {"tam": "PRF"}]
        elif profile.pos == "VERB" and profile.root_class == "STRONG_G":
            hyps += [
                {"tam": "PST", "person": 3, "gender": "M", "number": "SG"},
                {"tam": "PST", "person": 3, "gender": "M", "number": "PL"},
                {"tam": "PST", "person": 3, "gender": "F", "number": "PL"},
            ]
            for person, gender, number in (
                    (3, "M", "SG"), (3, "F", "SG"),
                    (3, "M", "PL"), (3, "F", "PL"),
                    (2, "M", "SG"), (2, "F", "SG"),
                    (2, "M", "PL"), (2, "F", "PL"),
                    (1, None, "SG"), (1, None, "PL")):
                hyps.append({
                    "tam": "SBJV", "person": person,
                    "gender": gender, "number": number})
            for png in ("3FS", "1S", "2MS", "2FS", "2MP", "2FP", "1P"):
                hyps.append({"tam": "PST", "png": png})
            for t in ("PRS", "PRF"):
                for person in (1, 2, 3):
                    for gender, number in (
                            ("M", "SG"), ("F", "SG"),
                            ("M", "PL"), ("F", "PL")):
                        hyps.append({
                            "tam": t, "person": person,
                            "gender": gender, "number": number})
                for png in ("FS", "MP", "FP"):
                    hyps.append({"tam": t, "png": png})
            for gender, number in (
                    ("M", "SG"), ("F", "SG"), ("M", "PL"), ("F", "PL")):
                hyps.append({"tam": "IPV", "person": 2,
                             "gender": gender, "number": number})
        elif profile.pos == "VERB" and (profile.root_class or "").startswith("DERIVED_"):
            hyps += [
                {"tam": "PST"},
                {"tam": "PST", "person": 3, "gender": "M", "number": "PL"},
                {"tam": "PST", "person": 3, "gender": "F", "number": "PL"},
                {"tam": "SBJV", "person": 1, "number": "SG"},
                {"tam": "SBJV", "person": 1, "number": "PL"},
                {"tam": "IPV"},
            ]
            for person in (2, 3):
                for number in ("SG", "PL"):
                    for gender in ("M", "F"):
                        hyps.append({
                            "tam": "SBJV", "person": person,
                            "gender": gender, "number": number,
                        })
            for png in ("3FS", "1S", "2MS", "2FS", "2MP", "2FP", "1P"):
                hyps.append({"tam": "PST", "png": png})
            for tam in ("PRS", "PRF"):
                for person in (1, 2, 3):
                    for gender, number in (
                            ("M", "SG"), ("F", "SG"),
                            ("M", "PL"), ("F", "PL")):
                        hyps.append({
                            "tam": tam, "person": person,
                            "gender": gender, "number": number,
                        })
                for png in ("FS", "MP", "FP"):
                    hyps.append({"tam": tam, "png": png})
        return hyps
