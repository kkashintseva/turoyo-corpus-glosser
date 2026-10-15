"""Current view of the append-only reusable-reading journal."""
import json
import unicodedata
from pathlib import Path

from maalula_glosser.io_utils import iter_jsonl
from maalula_glosser.state import state_root, writable


def surface_key(surface: str) -> str:
    """Match a word ignoring outer punctuation and case.

    A correction recorded for maplēlun must still fire for Maplēlun, for
    Maplēlun. at the end of a sentence and for mAplēlun (test_reusable_
    morphology_is_case_insensitive and test_live_source_alias_invalidates_
    cached_results_and_retraction_removes_both_keys in
    tests/integration/test_editor_corrections.py both depend on full
    case-folding here, not just the initial letter).

    Clitic boundaries are NOT folded away: ``ayṯul``, ``ayṯ-ul`` and
    ``ayṯ=ul`` stay three different keys, because a merge, a split or a
    cliticisation is a different token, not a different spelling of one
    (test_structural_surface_separators_are_preserved in
    tests/unit/test_surface_reuse_and_case.py, and the same rule in
    ``human_surface_key``).  A row typed with boundaries reaches the other
    spelling through ``original_surfaces`` instead: every recorded surface
    of a row is indexed on its own.
    """
    text = " ".join(unicodedata.normalize("NFC", surface).replace("‿", " ").split())
    def punctuation(char: str) -> bool:
        return unicodedata.category(char).startswith("P")
    while text and punctuation(text[0]):
        text = text[1:]
    while text and punctuation(text[-1]):
        text = text[:-1]
    return text.casefold()


def morphology_surfaces(row: dict) -> list[str]:
    """Both the reconstructed word and its explicitly recorded source forms."""
    return list(dict.fromkeys(s for s in [
        row.get("surface_canonical"), row.get("surface_original"),
        *row.get("original_surfaces", []),
    ] if s))


def morphology_readings(root: Path) -> list[dict]:
    """Recover source spellings for older rules from their saved editor tokens.

    Only a token still carrying that exact active reusable reading can supply
    an alias. Local edits and stale links left by Undo do not become rules.
    This read-time compatibility layer never rewrites documents or journals.
    """
    rows = active_readings(writable(root, "data/corrections_v2/reusable_morphology.jsonl"))
    by_id = {row.get("addition_id"): row for row in rows}
    for path in sorted(writable(root, "data/editor_sessions").glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        for sentence in document.get("sentences", []):
            for token in sentence.get("tokens", []):
                record = token.get("reusable_record") or {}
                row = by_id.get(record.get("addition_id"))
                if row is None or not row.get("explicit_reuse"):
                    continue
                cells = token.get("morphemes") or []
                segmentation = "".join(m.get("sep", "") + m.get("form", "") for m in cells)
                gloss = "".join(m.get("sep", "") + m.get("gloss", "") for m in cells)
                if segmentation != row.get("segmentation") or gloss != row.get("gloss"):
                    continue
                original = token.get("surface")
                if original and original not in morphology_surfaces(row):
                    row.setdefault("original_surfaces", []).append(original)
    return rows


def active_readings(path: Path) -> list[dict]:
    active = {}
    for index, row in enumerate(iter_jsonl(path)):
        if row.get("scope") == "retracted":
            active.pop(row.get("target_addition_id"), None)
        else:
            key = row.get("addition_id") or f"legacy:{index}"
            active.pop(key, None)
            active[key] = row
    return list(active.values())


def sync_journal(root: Path, relative: str) -> Path:
    """Merge checkout rows a server's already-seeded volume has not seen yet.

    `writable()` seeds a fresh volume once and then leaves it alone — right
    for pure server state, wrong for a curated journal like
    `reusable_morphology.jsonl`: an author-confirmed fix landed by `git push`
    (a rebuild, not a live edit) must still reach a volume that was seeded
    on an earlier boot. Rows are matched by `addition_id`; a checkout row the
    volume lacks is appended at the end, so it becomes the newest entry for
    its surface — the journal's own last-one-wins rule
    (`compositional.py::_from_reusable_morphology`) then has a git-delivered
    correction outrank whatever the volume already held.

    A no-op on a laptop, where the volume *is* the checkout.
    """
    path = writable(root, relative)
    source = Path(root) / relative
    if state_root(root).resolve() == Path(root).resolve() or not source.exists():
        return path
    known = {row.get("addition_id") for row in iter_jsonl(path)}
    new_rows = [row for row in iter_jsonl(source)
                if row.get("addition_id") not in known]
    if new_rows:
        with path.open("a", encoding="utf-8") as handle:
            for row in new_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path
