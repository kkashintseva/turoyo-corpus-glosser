"""Слова, которых нет ни в MASC, ни у Арнольда, но которые автор добавил сам.

Append-only журнал с отзывом через tombstone — то же соглашение, что у
`reusable_lexical_gloss.jsonl`. Здесь живёт только идентичность слова
(словарная форма, часть речи, набор форм); значение по-прежнему пишется
журналом значений, чтобы не было двух источников правды.

`lexeme_id` — это тот же `lexnew_...`, который токены уже носят, поэтому
продвижение не требует переписывать документы.
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any, Iterable

from maalula_glosser.gloss_shape import is_grammatical_label, looks_like_a_lemma
from maalula_glosser.io_utils import stable_id
from maalula_glosser.reusable import active_readings
from maalula_glosser.state import writable

LEXICON_ADDITIONS = "data/corrections_v2/lexicon_additions.jsonl"
PARTS_OF_SPEECH = frozenset(("verb", "adjective", "noun"))


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value or "")


def _path(root: Path | str) -> Path:
    return writable(Path(root), LEXICON_ADDITIONS)


def record_addition(root: Path | str, lexeme_id: str, lemma_canonical: str,
                    part_of_speech: str | None = None,
                    surfaces: Iterable[dict[str, Any]] = (),
                    titles: Iterable[str] = (),
                    root_canonical: str | None = None,
                    grammatical_info: str | None = None) -> dict[str, Any]:
    lexeme_id = (lexeme_id or "").strip()
    lemma = nfc(lemma_canonical).strip()
    if not lexeme_id:
        raise ValueError("lexeme_id is required")
    if lemma.startswith("["):
        raise ValueError("a corpus group is not a lexical identity")
    if is_grammatical_label(lemma) or not looks_like_a_lemma(lemma):
        raise ValueError("this is not a citation form")
    if part_of_speech is not None and part_of_speech not in PARTS_OF_SPEECH:
        raise ValueError(f"part_of_speech must be one of {sorted(PARTS_OF_SPEECH)} or null")
    surfaces_list = [{"form_canonical": nfc(s.get("form_canonical")),
                      "frequency": int(s.get("frequency") or 0)}
                     for s in surfaces if nfc(s.get("form_canonical"))]
    titles_set = sorted({t for t in titles if t})
    row = {
        "addition_id": stable_id("lexadd", lexeme_id),
        "lexeme_id": lexeme_id,
        "lemma_canonical": lemma,
        "part_of_speech": part_of_speech,
        "grammatical_info": nfc(grammatical_info or "").strip() or None,
        "surfaces": surfaces_list,
        "titles": titles_set,
        "scope": "lexicon_addition",
        "basis": "author-promoted from own text in the Словник",
    }
    root_text = nfc(root_canonical or "").strip().lstrip("√").strip()
    if root_text:
        # the root the author gave: the analyzer indexes the lexeme under it
        # (`lexeme_ids_by_root`), so its verbs/nouns find their relatives
        row["root_canonical"] = root_text
    path = _path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        matches = [r for r in active_readings(path)
                  if r.get("addition_id") == row["addition_id"]]
        if matches:
            last = matches[-1]
            unchanged = (
                last.get("lemma_canonical") == row["lemma_canonical"]
                and last.get("part_of_speech") == row["part_of_speech"]
                and last.get("grammatical_info") == row["grammatical_info"]
                and last.get("surfaces") == row["surfaces"]
                and last.get("titles") == row["titles"]
                and last.get("root_canonical") == row.get("root_canonical"))
            if unchanged:
                return last
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def extend_addition(root: Path | str, lexeme_id: str, lemma_canonical: str,
                    part_of_speech: str | None = None,
                    surface: str | None = None, title: str | None = None,
                    root_canonical: str | None = None,
                    grammatical_info: str | None = None) -> dict[str, Any]:
    """Create or extend an author lexeme without discarding earlier forms.

    Direct glossing of a new FLEX text can discover a lexeme one token at a
    time.  ``record_addition`` takes a complete snapshot, so calling it with
    only the current token would otherwise replace the previous surface/title
    list.  This helper merges the active snapshot first and is therefore safe
    to call repeatedly while the author works through a text.
    """
    root = Path(root)
    addition_id = stable_id("lexadd", lexeme_id)
    previous = next((row for row in reversed(active_additions(root))
                     if row.get("addition_id") == addition_id), None)
    counts: dict[str, int] = {}
    titles = set()
    if previous:
        for row in previous.get("surfaces") or []:
            form = nfc(row.get("form_canonical")).strip()
            if form:
                counts[form] = max(counts.get(form, 0),
                                   int(row.get("frequency") or 0))
        titles.update(t for t in previous.get("titles") or [] if t)
        if part_of_speech is None:
            part_of_speech = previous.get("part_of_speech")
        if not root_canonical:
            root_canonical = previous.get("root_canonical")
        if grammatical_info is None:
            grammatical_info = previous.get("grammatical_info")
    form = nfc(surface or "").strip()
    if form:
        # This is presence evidence, not a concordance counter: repeated edits
        # of one token must not inflate its frequency.  Exact corpus counts
        # live elsewhere and are still used when available.
        counts[form] = max(counts.get(form, 0), 1)
    if title:
        titles.add(title)
    surfaces = [{"form_canonical": form, "frequency": count}
                for form, count in sorted(counts.items())]
    return record_addition(
        root, lexeme_id, lemma_canonical, part_of_speech=part_of_speech,
        surfaces=surfaces, titles=sorted(titles), root_canonical=root_canonical,
        grammatical_info=grammatical_info)


def retract_addition(root: Path | str, addition_id: str) -> dict[str, Any]:
    from uuid import uuid4
    path = _path(root)
    if not any(row.get("addition_id") == addition_id
              for row in active_readings(path)):
        return {"target_addition_id": addition_id, "already_inactive": True}
    row = {"addition_id": f"retraction_{uuid4().hex}", "scope": "retracted",
           "target_addition_id": addition_id,
           "note": "Undone from the Словник"}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def active_additions(root: Path | str) -> list[dict[str, Any]]:
    return [row for row in active_readings(_path(root))
            if row.get("scope") == "lexicon_addition"]
