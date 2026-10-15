"""Gloss-label inventory: the vocabulary the editor autocompletes against.

The authoritative set lives in `config/gloss_inventory.json`, seeded from the
project's own sources (the abbreviation table in `glossed_all.docx` and the
FS_SHABO section-II convention). It has two parts:

- `labels`: the canonical display labels a glosser should actually type
  (SC, PC, NC, PRF, DO, FREE, person/gender/number bundles, …), each with a
  definition and a category;
- `display_map`: internal or legacy spellings folded into a canonical label
  (PRS→NC, SBJV→PC, SG→S, …). PST is a separate canonical label for the
  copular/pseudoverbal morpheme in `PST-be-…`. NC1 and NC2 are
  canonical display subclasses of the internal PRF TAM; unresolved perfects
  remain under PRF.

Autocomplete offers only the canonical labels, never a `display_map` source: a
label the program would immediately rewrite must not be suggested. The map is
still honoured when a cell is committed, and it lets the label window group a
form stored internally as PRS under the NC it displays as.

Editing this file — or the label window, which writes it — changes the labels
globally, so a rename survives a rebuild (unlike a hand-edit to generated data).
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from maalula_glosser.editor.senses import kwic
from maalula_glosser.state import writable
from maalula_glosser.reusable import surface_key

INVENTORY = "config/gloss_inventory.json"
# the confirmed corpus, source of the concordance shown in the label window
CORPUS = "data/final/analyses.jsonl"
# consulted only to migrate a project that predates the inventory file
LEGACY_CONFIG = "config/gloss_labels.json"


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


# Agreement labels: an optional person digit, an optional gender, an optional
# number. Matched case-insensitively so a lower-cased `3ms` is still recognised
# as a label — the upper-case heuristic alone cannot see it.
AGREEMENT_RE = re.compile(r"^[123]?[MF]?[SP]?$", re.IGNORECASE)


def _label_parts(gloss: str, known: set[str] | None = None) -> list[str]:
    """Grammatical labels in a gloss cell, lexical words dropped.

    `make.PRF` contributes `PRF`; `cool.down` contributes nothing.

    Three ways a component qualifies, because no single test covers the range:

    - it is in `known`, the inventory's own vocabulary, compared case-blind;
    - it looks like an agreement label (`3ms`, `MS`, `1S`, `2`) — this is what
      keeps the lower-cased gender labels visible;
    - it is upper case with at least one alphanumeric, the original heuristic,
      which still catches SC, DO, FREE and a bare person digit.

    A bare digit matters on its own: a person label with no gender or number is
    written `2`, as in `č-nufḏ-un` → `2-reach.SBJV-MP`.
    """
    out = []
    for component in gloss.replace("-", ".").replace("=", ".").replace("+", ".").split("."):
        token = component.strip()
        if not token or not any(c.isalnum() for c in token):
            continue
        if known is not None and token.casefold() in known:
            out.append(token)
        elif AGREEMENT_RE.match(token):
            out.append(token)
        elif token == token.upper():
            out.append(token)
    return out


# Labels whose spelling collides with an ordinary English word. `DO` is the
# direct-object marker, `do` is the verb, and the corpus writes both upper case.
AMBIGUOUS_WITH_LEXICAL = {"DO"}


def lexical_case(gloss: str, known: set[str] | None = None) -> str:
    """Lower-case a label that is standing in for an English word.

    `=DO=` is a whole morpheme and stays the grammatical marker. In `DO.PRF-F`
    the same spelling is the lexical stem of the verb, sitting where `make` or
    `flee` sits in every other analysis, so it is written `do.PRF-F`.

    Deciding by position rather than by meaning is what makes this safe: only a
    component sharing its morpheme with a category label is treated as lexical,
    so `mxass-l-il=le dress.PRF-DO-MP=3MS` keeps its marker.
    """
    text = nfc(gloss or "")
    if not text:
        return text
    out = []
    for morpheme in re.split(r"([-=+])", text):
        if morpheme in "-=+" or not morpheme:
            out.append(morpheme)
            continue
        parts = morpheme.split(".")
        if len(parts) > 1:
            parts = [p.lower() if p.upper() in AMBIGUOUS_WITH_LEXICAL else p
                     for p in parts]
        out.append(".".join(parts))
    return "".join(out)


def _merge_inventory(volume: dict, checkout: dict) -> dict:
    """A rename or definition landed by `git push`, folded into a volume seeded
    on an earlier boot.

    `labels`, `display_map` and `case_map` are curated dicts, editable both in
    the checkout (a commit) and live (the "Инвентарь глосс" window, which
    writes to the volume). Merged key-wise so neither source loses entries the
    other lacks; the volume wins a key both define, since that is the one
    being edited live. Every other top-level field (`_comment`, `authority`,
    `_case_rule`, …) is metadata, not editable at runtime: the volume's value
    if it has one, else the checkout's.
    """
    merged = {**checkout, **volume}
    for key in ("labels", "display_map", "case_map"):
        merged[key] = {**checkout.get(key, {}), **volume.get(key, {})}
    return merged


class LabelInventory:
    def __init__(self, root: Path | str = ".", *,
                 inventory_path: str = INVENTORY,
                 migrate_legacy: bool = True):
        self.root = Path(root)
        self.inventory_path = inventory_path
        self.migrate_legacy = migrate_legacy
        self.labels: dict[str, dict[str, str]] = {}
        self.display_map: dict[str, str] = {}

        # Orthography, kept apart from display_map on purpose. display_map
        # substitutes one label for another (PRS→NC) and must not be applied to
        # analyzer output. PST is deliberately not in display_map: the past
        # copula/pseudoverb uses it as an independent morpheme.
        # case_map only changes how the same label is written (MS→ms), so it is
        # safe to apply everywhere, including to what the glosser emits.
        self.case_map: dict[str, str] = {}

        path = writable(self.root, self.inventory_path)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            checkout_path = self.root / INVENTORY
            if checkout_path.resolve() != path.resolve() and checkout_path.exists():
                data = _merge_inventory(
                    data, json.loads(checkout_path.read_text(encoding="utf-8")))
            self.labels = {nfc(k): dict(v) for k, v in data.get("labels", {}).items()}
            self.display_map = {nfc(k).upper(): nfc(v).upper()
                                for k, v in data.get("display_map", {}).items()}
            self.case_map = {nfc(k).casefold(): nfc(v)
                             for k, v in data.get("case_map", {}).items()}
        elif self.migrate_legacy:
            self._migrate_from_legacy()

    def _migrate_from_legacy(self) -> None:
        """Fall back to the old config so a fresh checkout still autocompletes.

        Only the alias sources are dropped from the suggestion pool; without the
        curated definitions every label simply has an empty one.
        """
        legacy = self.root / LEGACY_CONFIG
        if not legacy.exists():
            return
        data = json.loads(legacy.read_text(encoding="utf-8"))
        aliases = {nfc(k).upper(): nfc(v).upper()
                   for k, v in data.get("aliases", {}).items()}
        self.display_map = {**aliases, "PRS": "NC", "SBJV": "PC"}
        notes = {nfc(k): v for k, v in data.get("notes", {}).items()}
        for label in data.get("seed_labels", []):
            canonical = self.canonical(label)
            if canonical.upper() in {s.upper() for s in self.display_map}:
                continue
            self.labels.setdefault(canonical, {"definition": notes.get(canonical, ""),
                                               "category": "other"})

    # ---- case ----
    def cased(self, label: str | None) -> str:
        """The label written the way the project writes it (`MS` → `ms`).

        Case only. `SC` stays `SC`; nothing is substituted for anything else.
        """
        text = nfc(label or "")
        return self.case_map.get(text.casefold(), text)

    def known(self) -> set[str]:
        """Every label the inventory recognises, case-folded, for `_label_parts`."""
        vocabulary = set(self.labels) | set(self.display_map) | set(self.case_map)
        vocabulary |= set(self.case_map.values())
        return {nfc(x).casefold() for x in vocabulary}

    def apply_case(self, gloss: str | None) -> str:
        """Rewrite the grammatical components of a gloss into project casing.

        Applied to analyzer output as well as to typed input: the glosser emits
        `3MS`, the project writes `3ms`, and that difference is orthographic, so
        it does not need the confirmation a substitution would.
        """
        text = nfc(gloss or "")
        if not text:
            return text
        pieces = re.split(r"([-=+.])", text)
        known = self.known()
        out = []
        for piece in pieces:
            if piece in "-=+." or not piece:
                out.append(piece)
                continue
            if piece.casefold() in known or AGREEMENT_RE.match(piece):
                out.append(self.cased(piece))
            else:
                out.append(piece)
        # `DO` standing in for the English verb is lexical and written `do`
        return lexical_case("".join(out), known)

    # ---- folding ----
    def canonical(self, label: str | None) -> str:
        """The display label a spelling folds to (`PRS` → `NC`); identity else.

        Casing is applied too, so a user typing `3MS` lands on `3ms`.
        """
        text = nfc(label or "")
        folded = self.display_map.get(text.upper(), text)
        return self.cased(folded)

    def registered(self) -> list[str]:
        """The canonical labels, sorted. This is the autocomplete vocabulary."""
        return sorted(self.labels)

    def definition_for(self, label: str) -> str | None:
        """What the label means — the tooltip text under an autocomplete row."""
        entry = self.labels.get(self.canonical(label))
        return (entry or {}).get("definition") or None

    def note_for(self, label: str) -> str | None:
        """A caveat attached to the label, distinct from its definition.

        Only a handful carry one, and they exist to stop plausible-looking
        blanket edits. A rename must show this before applying.
        """
        entry = self.labels.get(self.canonical(label))
        return (entry or {}).get("note") or None

    def suggest(self, prefix: str, extra: list[str] | None = None,
                limit: int = 8) -> list[str]:
        """Canonical labels containing `prefix`, prefix matches ranked first.

        Substring, not just leading: the user asked for candidates "containing
        the same letters". `extra` (labels already in the document) is folded to
        canonical form and never reintroduces a legacy spelling.
        """
        needle = nfc(prefix).upper()
        pool = list(dict.fromkeys(
            self.registered()
            + [self.canonical(x) for x in extra or []
               if self.canonical(x).upper() not in self.display_map]))
        if not needle:
            return sorted(pool)[:limit]
        starts = [x for x in pool if x.upper().startswith(needle)]
        inside = [x for x in pool if needle in x.upper() and x not in starts]
        return (sorted(starts) + sorted(inside))[:limit]

    # ---- editing the inventory ----
    def register(self, label: str, *, category: str = "corpus") -> bool:
        """Add a corpus-discovered label without overwriting human metadata."""
        canonical = nfc(label).strip()
        if not canonical or canonical in self.labels:
            return False
        self.labels[canonical] = {"definition": "", "category": category}
        return True

    def save(self) -> Path:
        """Write the inventory back to its curated config.

        This is why a rename is global and survives a rebuild: the file is an
        input to the pipeline, not generated output.
        """
        path = writable(self.root, self.inventory_path)
        data = {}
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        data["labels"] = self.labels
        data["display_map"] = dict(sorted(self.display_map.items()))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        return path

    def set_definition(self, label: str, text: str) -> None:
        canonical = self.canonical(label)
        if canonical not in self.labels:
            raise KeyError(canonical)
        self.labels[canonical]["definition"] = nfc(text).strip()

    def rename(self, old: str, new: str) -> dict[str, Any]:
        """Rename a label everywhere, merging if the target already exists.

        The old spelling is kept as a `display_map` source rather than deleted,
        so a document, corpus row or hand-typed cell still written the old way
        folds to the new label instead of becoming an unknown one. Anything that
        previously folded to `old` is repointed at `new` in the same pass — that
        is what stops a chain like PRS→NC→X from silently breaking when NC is
        renamed.
        """
        source, target = self.canonical(old), nfc(new).strip()
        if not target:
            raise ValueError("new label is empty")
        if source not in self.labels:
            raise KeyError(source)
        if target == source:
            return {"renamed": False, "merged": False}

        entry = self.labels.pop(source)
        merged = target in self.labels
        if not merged:
            self.labels[target] = entry
        self.display_map[source.upper()] = target.upper()
        for key, value in list(self.display_map.items()):
            if value == source.upper():
                self.display_map[key] = target.upper()
        return {"renamed": True, "merged": merged}

    def inventory_view(self, usage: dict[str, list] | None = None) -> list[dict]:
        """The canonical labels for the label window, grouped by category.

        Each row carries its definition, the legacy spellings that fold into it,
        whether it still needs a definition, and (if given) its concordance.
        """
        folded: dict[str, list[str]] = {}
        for source, target in self.display_map.items():
            folded.setdefault(target, []).append(source)
        rows = []
        for label in self.registered():
            entry = self.labels[label]
            rows.append({
                "label": label,
                "definition": entry.get("definition", ""),
                "note": entry.get("note", ""),
                "category": entry.get("category", "other"),
                "folds_in": sorted(folded.get(label.upper(), [])),
                "needs_definition": not entry.get("definition"),
                "usage": (usage or {}).get(label, []),
            })
        rows.sort(key=lambda r: (r["category"], r["label"]))
        return rows


def corpus_rows(root: Path | str = ".",
                analyzer=None) -> list[dict[str, str]]:
    """Every confirmed corpus analysis, as the program actually reads it.

    Not the raw JSONL. `data/final/analyses.jsonl` still stores glosses annotated
    under the cancelled blanket NC→PRF rule, and the analyzer repairs them from
    `corrections_v2/gold_nc_prs_corrections.jsonl` on the way in, drops excluded
    triples and lets the authoritative IGT overlay win. Reading the file directly
    reproduces exactly the stale readings the pipeline exists to correct — that
    is how `zel-le` surfaced as `DO.PRF-3MS` when the program itself says
    `go.NC-3ms`.

    So the source is the analyzer's own trusted-corpus index, rendered through
    the same display convention the editor shows elsewhere.
    """
    from maalula_glosser.compositional import CompositionalAnalyzer
    from maalula_glosser.normalization.gloss import render_authoritative_gloss
    from maalula_glosser.morphology.perfect_subclass import (
        classify_prf_analysis, replace_prf_label,
    )

    if analyzer is None:
        analyzer = CompositionalAnalyzer(Path(root))
    rows = []
    # A reusable correction is the curated reading for that surface and must
    # replace the stale source-corpus row in the inventory.  Previously the
    # correction endpoint rebuilt the analyzer, but this function still read
    # only ``analyzer.a1``; the corrected row therefore vanished on reload.
    surfaces = set(analyzer.a1) | set(analyzer.reusable_morphology)
    for surface in sorted(surfaces):
        corrected = analyzer.reusable_morphology.get(surface_key(surface)) or []
        analyses = corrected if corrected else analyzer.a1.get(surface, [])
        for analysis in analyses:
            segmentation = nfc(analysis.get("segmentation") or "")
            gloss = nfc(analysis.get("gloss_clean") or analysis.get("gloss") or "")
            if not segmentation or not gloss:
                continue
            # The concordance reads the trusted-corpus index directly rather
            # than calling `_from_a1`, so apply the same conservative display
            # subclassification here. Internal/corpus TAM remains PRF.
            if getattr(analyzer, "profile", None) == "flex":
                display_gloss = gloss
            else:
                lexeme = analyzer.lexemes.get(analysis.get("lemma_id"))
                perfect = classify_prf_analysis(lexeme, segmentation, gloss)
                if perfect:
                    gloss = replace_prf_label(gloss, perfect[0])
                display_gloss = render_authoritative_gloss(gloss)
            rows.append({"surface": surface, "segmentation": segmentation,
                         "gloss": display_gloss})
    return rows


def session_concordance(store, inventory: "LabelInventory | None" = None,
                        limit: int | None = None
                        ) -> dict[str, list[dict[str, Any]]]:
    """Canonical label → readings currently present in saved editor texts.

    Saved texts are editable project data, not a disposable UI cache.  They
    therefore belong in the inventory alongside the confirmed corpus.  Only
    committed glosses are counted: an unaccepted suggestion must not silently
    become part of the project's label vocabulary.
    """
    from maalula_glosser.editor.session import join_parts

    fold = inventory.canonical if inventory else (lambda x: x)
    known = inventory.known() if inventory else None
    out: dict[str, list[dict[str, Any]]] = {}
    seen: dict[str, set[tuple[str, str, str]]] = {}
    for listed in store.list():
        document = store.load(listed["document_id"])
        for sentence in document.get("sentences") or []:
            for token in sentence.get("tokens") or []:
                morphemes = token.get("morphemes") or []
                segmentation = join_parts(morphemes)
                gloss = join_parts(morphemes, key="gloss")
                if not segmentation or not gloss:
                    continue
                if inventory is not None:
                    gloss = inventory.apply_case(gloss)
                surface = nfc(token.get("normalized") or token.get("surface") or "")
                if not surface:
                    continue
                row = {
                    "surface": surface,
                    "segmentation": segmentation,
                    "gloss": gloss,
                    "source": "text",
                    "document_id": document.get("document_id"),
                    "token_id": token.get("token_id"),
                    "title": document.get("title") or "(без названия)",
                    "user_edited": bool(token.get("locked") or any(
                        morpheme.get("origin") == "user" for morpheme in morphemes)),
                }
                key = (surface.casefold(), segmentation, gloss)
                for label in {fold(p) for p in _label_parts(gloss, known)}:
                    bucket = out.setdefault(label, [])
                    marks = seen.setdefault(label, set())
                    if key in marks or (limit is not None and len(bucket) >= limit):
                        continue
                    marks.add(key)
                    bucket.append(row)
    return out


def combined_concordance(root: Path | str, store,
                         inventory: "LabelInventory | None" = None,
                         limit: int | None = None, analyzer=None
                         ) -> dict[str, list[dict[str, Any]]]:
    """Merge saved-text and confirmed-corpus readings without duplicates.

    Text rows come first, so an author's current correction is visible before
    the older corpus attestation that prompted it.  A reusable correction still
    replaces the stale corpus row entirely via :func:`corpus_rows`.
    """
    text = session_concordance(store, inventory, limit=None)
    corpus = corpus_concordance(root, inventory, limit=None, analyzer=analyzer)
    labels = set(text) | set(corpus)
    merged: dict[str, list[dict[str, Any]]] = {}
    for label in labels:
        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for row in text.get(label, []) + corpus.get(label, []):
            item = dict(row)
            item.setdefault("source", "corpus")
            key = (nfc(item.get("surface") or "").casefold(),
                   nfc(item.get("segmentation") or ""),
                   nfc(item.get("gloss") or ""))
            if key in seen:
                continue
            seen.add(key)
            rows.append(item)
            if limit is not None and len(rows) >= limit:
                break
        merged[label] = rows
    return merged


def corpus_concordance(root: Path | str = ".",
                       inventory: "LabelInventory | None" = None,
                       limit: int | None = None, analyzer=None
                       ) -> dict[str, list[dict[str, str]]]:
    """canonical label → attested `{surface, segmentation, gloss}` rows.

    The window asks for "every word in the database carrying this gloss", so the
    source is the whole confirmed corpus, not just the open document. Rows fold
    to canonical labels, so a form stored as PRS is listed under the NC it
    displays as.

    Exhaustive by default: the window is a research tool, and a concordance that
    silently stops at the first n rows would misrepresent how a label is used.
    `limit` exists for callers that genuinely want a sample.
    """
    fold = inventory.canonical if inventory else (lambda x: x)
    known = inventory.known() if inventory else None
    out: dict[str, list[dict[str, str]]] = {}
    seen: dict[str, set[tuple[str, str]]] = {}
    for row in corpus_rows(root, analyzer):
        gloss = row["gloss"]
        # the corpus predates the casing convention and stores 3MS; show it the
        # way the project writes it, so the window does not contradict the
        # inventory entry the reader just clicked
        if inventory is not None:
            gloss = inventory.apply_case(gloss)
            gloss = lexical_case(gloss, known)
        entry = {"surface": row["surface"],
                 "segmentation": row["segmentation"], "gloss": gloss}
        key = (row["segmentation"], gloss)
        for label in {fold(p) for p in _label_parts(gloss, known)}:
            bucket = out.setdefault(label, [])
            marks = seen.setdefault(label, set())
            if key in marks or (limit is not None and len(bucket) >= limit):
                continue
            marks.add(key)
            bucket.append(entry)
    return out


CORRECTIONS = "data/corrections_v2/reusable_morphology.jsonl"


def record_correction(root: Path | str, surface: str, segmentation: str,
                      gloss: str, note: str = "", *,
                      lemma_id: str | None = None,
                      original_surface: str | None = None) -> dict[str, Any]:
    """Record an author-confirmed reading for a surface, applied everywhere.

    Written to `corrections_v2/reusable_morphology.jsonl`, the curated layer the
    analyzer already reads as tier A1. That is the whole point of putting it
    there rather than editing `data/final`: the fix reaches every text that
    contains the form, and a rebuild cannot silently undo it.

    The corpus row that prompted the edit is left alone. It stays as evidence of
    what the source document said; the correction outranks it.
    """
    from maalula_glosser.io_utils import stable_id

    root = Path(root)
    surface, segmentation, gloss = (nfc(surface).strip(),
                                    nfc(segmentation).strip(), nfc(gloss).strip())
    if not surface or not segmentation or not gloss:
        raise ValueError("surface, segmentation and gloss are all required")
    bare = segmentation.replace("-", "").replace("=", "").replace("+", "")
    if bare != surface:
        raise ValueError(
            f"segmentation {segmentation!r} does not reconstruct {surface!r}")
    if len(re.split(r"[-=+]", segmentation)) != len(re.split(r"[-=+]", gloss)):
        raise ValueError("segmentation and gloss have different morpheme counts")
    if re.findall(r"[-=+]", segmentation) != re.findall(r"[-=+]", gloss):
        raise ValueError("segmentation and gloss have different morpheme boundaries")
    if "?" in gloss or "*" in gloss:
        raise ValueError("an uncertain gloss cannot become a reusable correction")

    row = {
        "addition_id": stable_id("rm", surface, segmentation, gloss),
        "surface_canonical": surface,
        "segmentation": segmentation,
        "gloss": gloss,
        "scope": "reusable",
        "explicit_reuse": True,
        "basis": "author-confirmed in the gloss editor",
        "note": note or "recorded from the label inventory window",
        "lemma_id": lemma_id,
    }
    if original_surface:
        row["surface_original"] = nfc(original_surface).strip()
        # Distinct source spellings must not erase each other's aliases.
        if surface_key(original_surface) != surface_key(surface):
            row["addition_id"] = stable_id(
                "rm", surface, segmentation, gloss, surface_key(original_surface))
    path = writable(root, CORRECTIONS)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The same reading recorded twice is not news. Confirming a word the editor
    # already knows is an ordinary thing to do — the file would otherwise grow a
    # row per keystroke-ending edit, and the analyzer a duplicate candidate.
    # But a genuinely new lemma_id (the prior active row had None, or a
    # different lexeme_id) is real news — it must not be swallowed here, or a
    # later-discovered dictionary link could never reach the journal.
    if path.exists():
        from maalula_glosser.reusable import active_readings
        readings = [previous for previous in active_readings(path)
                    if previous.get("surface_canonical") == surface]
        if (readings and readings[-1].get("addition_id") == row["addition_id"]
                and readings[-1].get("explicit_reuse")
                and readings[-1].get("lemma_id") == lemma_id
                and (not original_surface
                     or readings[-1].get("surface_original") == row["surface_original"])):
            return readings[-1]
        # Reconfirming an older alternative changes its priority. Record that
        # ordering too, or a restart would resurrect a different latest reading.
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def backfill_reusable_lemma_ids(root: Path | str, analyzer) -> dict[str, Any]:
    """Give every already-active, `lemma_id`-less reusable reading a chance
    to pick up the dictionary link it should always have carried.

    A reading recorded before this journal tracked `lemma_id` (or recorded
    through a path that had no lexeme to hand) can still often be resolved
    now, the same way a fresh correction would be: run its own surface,
    segmentation and gloss back through the resolver. A unique hit is
    recorded as a new, superseding row (invariant #7 forbids guessing when
    the resolver itself reports more than one owner — those are left alone,
    not auto-picked). A row that already carries a `lemma_id` is not
    re-examined; it is already fine.

    One-time, explicitly-triggered operation — never run from an automatic
    startup/rebuild path.
    """
    from maalula_glosser.editor.session import resolve_reading
    from maalula_glosser.reusable import active_readings

    path = writable(root, CORRECTIONS)
    examined = 0
    upgraded = 0
    still_unresolved = 0
    for row in active_readings(path):
        if row.get("scope") != "reusable" or row.get("lemma_id"):
            continue
        examined += 1
        found = resolve_reading(row.get("surface_canonical") or "",
                                row.get("segmentation"), row.get("gloss"),
                                analyzer)
        if found is not None and found.lexeme_id and not found.candidates:
            record_correction(root, row["surface_canonical"], row["segmentation"],
                              row["gloss"],
                              note="backfilled: lemma_id recovered",
                              lemma_id=found.lexeme_id)
            upgraded += 1
        else:
            still_unresolved += 1
    return {"examined": examined, "upgraded": upgraded,
            "still_unresolved": still_unresolved}


def retract_correction(root: Path, addition_id: str) -> dict:
    """Retract one reading, preserving both the old decision and local text."""
    from uuid import uuid4
    from maalula_glosser.reusable import active_readings
    path = writable(root, CORRECTIONS)
    if not any(row.get("addition_id") == addition_id for row in active_readings(path)):
        return {"target_addition_id": addition_id, "already_inactive": True}
    row = {"addition_id": f"retraction_{uuid4().hex}", "scope": "retracted",
           "target_addition_id": addition_id,
           "note": "Explicitly undone from the originating editor token"}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


IGT_EXAMPLES = "data/final/igt_examples.jsonl"


def find_corpus_sentences(root: Path | str, surface: str,
                          limit: int = 12) -> list[dict[str, Any]]:
    """Corpus sentences containing a form, so a text can be opened from one.

    A form the label window lists may be in no open document at all — it is the
    corpus talking, not the user's texts. `data/final/igt_examples.jsonl` keeps
    the transcription and its translation next to the glossed tokens, so a
    sentence is enough to start editing from.
    """
    root = Path(root)
    needle = nfc(surface).strip().casefold()
    bare = needle.replace("-", "").replace("=", "").replace("+", "")
    path = root / IGT_EXAMPLES
    if not path.exists():
        return []
    hits = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        example = json.loads(line)
        surfaces = [nfc(token.get("surface_canonical") or "")
                    for token in example.get("tokens") or []]
        hit_index = None
        for position, token in enumerate(example.get("tokens") or []):
            for key in ("surface_canonical", "isolated_form", "segmentation"):
                value = nfc(token.get(key) or "").casefold()
                if not value:
                    continue
                bare_value = value.replace("-", "").replace("=", "").replace("+", "")
                if needle in (value, bare_value) or bare in (value, bare_value):
                    hit_index = position
                    break
            if hit_index is not None:
                break
        if hit_index is None:
            continue
        transcription = nfc(example.get("transcription_original") or "")
        if not transcription:
            continue
        before, after = kwic(surfaces, hit_index)
        hits.append({
            "example_id": example.get("example_id"),
            "source_text": example.get("source_text") or "",
            "transcription": transcription,
            "translation": nfc(example.get("translation_en")
                               or example.get("translation_original") or ""),
            "marker": example.get("segment_marker") or "",
            "context_before": before,
            "context_after": after,
        })
        if len(hits) >= limit:
            break
    return hits


IGT_SEGMENTS = "data/final/igt_segments.jsonl"


def corpus_texts(root: Path | str) -> list[dict[str, Any]]:
    """The whole texts the corpus holds, with enough detail to choose one."""
    root = Path(root)
    path = root / IGT_SEGMENTS
    if not path.exists():
        return []
    by_text: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        segment = json.loads(line)
        name = segment.get("source_text") or "(без названия)"
        entry = by_text.setdefault(name, {"source_text": name, "segments": 0,
                                          "characters": 0})
        entry["segments"] += 1
        entry["characters"] += sum(len(t) for t in
                                   segment.get("transcription_turns") or [])
    return sorted(by_text.values(), key=lambda e: -e["characters"])


def corpus_text_lines(root: Path | str, source_text: str
                      ) -> tuple[list[str], list[str]]:
    """A whole corpus text as parallel lines and translations, in document order.

    Turns are paired with their translations, and each turn is split into
    sentences. Pairing happens only where the counts match; where they do not,
    the turn's whole translation goes on its first line, because a plausible
    misalignment is worse than an obviously unsplit one.
    """
    from maalula_glosser.gloss_text import split_sentences

    root = Path(root)
    path = root / IGT_SEGMENTS
    if not path.exists():
        return [], []
    lines: list[str] = []
    translations: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        segment = json.loads(raw)
        if (segment.get("source_text") or "") != source_text:
            continue
        turns = segment.get("transcription_turns") or []
        turn_translations = segment.get("translation_turns_en") or []
        for index, turn in enumerate(turns):
            turn = nfc(turn).strip()
            if not turn:
                continue
            translation = nfc(turn_translations[index]
                              if index < len(turn_translations) else "").strip()
            pieces = split_sentences(turn) or [turn]
            parts = split_sentences(translation) if translation else []
            if len(parts) != len(pieces):
                parts = [translation] + [""] * (len(pieces) - 1)
            lines.extend(pieces)
            translations.extend(parts)
    return lines, translations


def corpus_sentence(root: Path | str, example_id: str) -> dict[str, Any] | None:
    root = Path(root)
    path = root / IGT_EXAMPLES
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or example_id not in line:
            continue
        example = json.loads(line)
        if example.get("example_id") != example_id:
            continue
        return {
            "example_id": example_id,
            "transcription": nfc(example.get("transcription_original") or ""),
            "translation": nfc(example.get("translation_en")
                               or example.get("translation_original") or ""),
            "marker": example.get("segment_marker") or "",
        }
    return None


def find_occurrences(store, surface: str) -> list[dict[str, Any]]:
    """Where a surface occurs in the saved sessions, for jumping to the text.

    Matched on the token as the analyzer saw it and on the written form, so a
    search for `zelle` finds the token whether the text spells it with typed
    boundaries or without.
    """
    needle = nfc(surface).strip().casefold()
    bare = needle.replace("-", "").replace("=", "").replace("+", "")
    hits = []
    for row in store.list():
        document = store.load(row["document_id"])
        for sentence in document.get("sentences") or []:
            for token in sentence.get("tokens") or []:
                candidates = {
                    nfc(token.get("surface") or "").casefold(),
                    nfc(token.get("normalized") or "").casefold(),
                    nfc(join_parts(token.get("morphemes") or [])).casefold(),
                }
                candidates |= {c.replace("-", "").replace("=", "").replace("+", "")
                               for c in candidates}
                if needle in candidates or bare in candidates:
                    hits.append({
                        "document_id": document["document_id"],
                        "title": document.get("title") or "(без названия)",
                        "sentence_id": sentence.get("sentence_id"),
                        "token_id": token.get("token_id"),
                        "surface": token.get("surface"),
                        "line": sentence.get("source_line"),
                    })
    return hits


def join_parts(morphemes) -> str:
    from maalula_glosser.editor.session import join_parts as _join

    return _join(morphemes)


def rewrite_documents(store, old: str, new: str) -> int:
    """Apply a rename to every stored session; returns the number changed.

    A global rename that left open documents on the old label would be a rename
    in name only, so the sessions are rewritten in the same operation. Only the
    grammatical component is replaced — `make.PST` becomes `make.SC` while a
    lexical `past` is untouched.
    """
    changed = 0
    for row in store.list():
        document = store.load(row["document_id"])
        touched = False
        for sentence in document.get("sentences") or []:
            for token in sentence.get("tokens") or []:
                for morpheme in token.get("morphemes") or []:
                    for field in ("gloss", "suggested"):
                        value = nfc(morpheme.get(field) or "")
                        if not value:
                            continue
                        replaced = _replace_label(value, old, new)
                        if replaced != value:
                            morpheme[field] = replaced
                            touched = True
        if touched:
            store.save(document)
            changed += 1
    return changed


def _replace_label(gloss: str, old: str, new: str) -> str:
    """Swap one grammatical component, leaving lexical material alone."""
    out, buffer, result = [], "", []
    for char in gloss:
        if char in "-=+.":
            out.append(buffer)
            out.append(char)
            buffer = ""
        else:
            buffer += char
    out.append(buffer)
    for piece in out:
        if piece == old:
            result.append(new)
        else:
            result.append(piece)
    return "".join(result)


def document_labels(document: dict[str, Any],
                    inventory: "LabelInventory | None" = None) -> list[str]:
    """Every grammatical label the document currently uses, sorted."""
    known = inventory.known() if inventory else None
    found: set[str] = set()
    for sentence in document.get("sentences") or []:
        for token in sentence.get("tokens") or []:
            for morpheme in token.get("morphemes") or []:
                found.update(_label_parts(nfc(morpheme.get("gloss") or ""), known))
                found.update(_label_parts(nfc(morpheme.get("suggested") or ""), known))
    return sorted(found)


def label_usage(document: dict[str, Any],
                inventory: "LabelInventory | None" = None
                ) -> dict[str, list[dict[str, str]]]:
    """canonical label → the tokens that use it, for the concordance.

    Each entry is one token as `{segmentation, gloss}`, so the window can print
    `morphemes gloss; morphemes gloss …` with the label highlighted. When an
    inventory is given, a form stored as an internal label (PRS) is grouped
    under the canonical one it displays as (NC).
    """
    from maalula_glosser.editor.session import join_parts

    fold = inventory.canonical if inventory else (lambda x: x)
    known = inventory.known() if inventory else None
    usage: dict[str, list[dict[str, str]]] = {}
    for sentence in document.get("sentences") or []:
        for token in sentence.get("tokens") or []:
            morphemes = token.get("morphemes") or []
            labels = set()
            for morpheme in morphemes:
                labels.update(fold(p) for p in _label_parts(nfc(morpheme.get("gloss") or ""), known))
            if not labels:
                continue
            row = {"segmentation": join_parts(morphemes),
                   "gloss": join_parts(morphemes, key="gloss")}
            for label in labels:
                usage.setdefault(label, []).append(row)
    return usage
