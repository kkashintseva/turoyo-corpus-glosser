"""Найти для слова из текста уже существующую лексему.

Порядок: засвидетельствованная форма → основа как форма → основа в
`stems_by_lexeme` → часть словарной формы (и она же без швы) → статья
Арнольда по написанию. На каждом шаге совпадение принимается только если
ведёт к одной лексеме; несколько владельцев — это выбор для автора, а не
повод привязать к первому (инвариант #7). По корню и по подстроке не ищем
никогда: `awʕ` похоже на `awʕeẓ yawʕeẓ` (корень wʕẓ), а правильный ответ —
`awəʕ yawəʕ` (корень wʕy).
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value or "")


@dataclass(frozen=True)
class Resolved:
    lexeme_id: str | None
    lemma: str
    gloss: str | None
    status: str
    basis: str
    candidates: tuple[str, ...] = field(default=())


def _owners(analyzer, key: str) -> set[str]:
    return {form.get("lexeme_id") for form
            in analyzer.forms_by_surface.get(key, [])
            if form.get("lexeme_id")}


def _lemma_of(analyzer, lexeme_id: str) -> str:
    return nfc((analyzer.lexemes.get(lexeme_id) or {}).get("lemma_canonical"))


def _resolved(analyzer, lexeme_id: str, basis: str) -> Resolved:
    gloss, status = analyzer.lexical.resolve(lexeme_id)
    return Resolved(lexeme_id=lexeme_id, lemma=_lemma_of(analyzer, lexeme_id),
                    gloss=gloss, status=status, basis=basis)


def resolve_lexeme(analyzer, surface: str, stem: str | None) -> Resolved | None:
    surface, stem = nfc(surface), nfc(stem or "")
    tiers: list[tuple[set[str], str]] = [
        (_owners(analyzer, surface), "attested form"),
        (_owners(analyzer, stem) if stem else set(), "attested stem form"),
        ({lexeme_id for lexeme_id, stems in analyzer.stems_by_lexeme.items()
          if lexeme_id and stem and stem in stems}, "analysed stem"),
    ]
    for key in (surface, stem):
        if key:
            tiers.append((set(analyzer.principal_parts_index.get(key, ())),
                          "principal part of the citation form"))
    for owners, basis in tiers:
        if len(owners) == 1:
            return _resolved(analyzer, next(iter(owners)), basis)
        if len(owners) > 1:
            return Resolved(lexeme_id=None, lemma="", gloss=None,
                            status="ambiguous", basis=basis,
                            candidates=tuple(sorted(owners)))
    for key in (surface, stem):
        for entry in (analyzer.lexicon_v3.get(key) or []) if key else ():
            headword = nfc(entry.get("headword"))
            if headword:
                # Арнольд даёт словарную форму и статью, но не часть речи и
                # не значение (инвариант #9).
                return Resolved(lexeme_id=None, lemma=headword, gloss=None,
                                status="arnold_headword", basis="Arnold entry")
    return None


def rewrite_minted_lexemes(store, analyzer) -> dict:
    """Переписать токены с минтованным `lexnew_...` там, где лексикон уже
    знает слово. Стем берётся, только если ровно одна ячейка морфологии несёт
    лексическую часть (`has_lexical_part`) — то же правило "ровно одна
    лексическая ячейка", что и `session._lexical_stem`; при нуле или
    нескольких ничего не выдумываем, иначе сегментация целиком снова
    становится леммой (так `apt-aṯ` и попал в заголовки).

    Идемпотентно по построению: считаем изменившимся только токен, у которого
    реально поменялось хоть одно из полей `lemma_id`/`lemma`/`lexical_gloss`/
    `lexeme_candidates`. Статья Арнольда без лексемы (`arnold_headword`) не
    трогает `lemma_id` — привязывать не к чему, — но обновляет `lemma` на
    цитатную форму; неоднозначный резолв (`ambiguous`) тоже не трогает
    `lemma_id` — выбор за автором (инвариант #7) — но записывает
    `lexeme_candidates`, иначе документ, сохранённый до резолвера, остался
    бы без этого выбора навсегда. Без сравнения "было/стало" по каждому из
    этих полей токен, сохранив свой `lexnew_...`, пересматривался бы на
    каждом прогоне заново."""
    from maalula_glosser.gloss_shape import has_lexical_part, looks_like_a_lemma

    by_document, total = [], 0
    for listed in store.list():
        document = store.load(listed["document_id"])
        touched = 0
        for sentence in document.get("sentences") or []:
            for token in sentence.get("tokens") or []:
                lexeme_id = token.get("lemma_id") or ""
                if not lexeme_id.startswith("lexnew_"):
                    continue
                lexical = [cell for cell in token.get("morphemes") or []
                           if has_lexical_part(cell.get("gloss") or "")]
                stem = (nfc(lexical[0].get("form") or "")
                        if len(lexical) == 1 else None)
                found = resolve_lexeme(analyzer, nfc(token.get("surface") or ""), stem)
                if found is None:
                    continue
                before = (token.get("lemma_id"), token.get("lemma"),
                          token.get("lexical_gloss"),
                          tuple(token.get("lexeme_candidates") or ()))
                if found.lexeme_id:
                    token["lemma_id"] = found.lexeme_id
                    token["lemma"] = found.lemma
                    # Авторская глосса важнее корпусной: её ввели вручную при
                    # импорте, и весь план держит её как первоисточник смысла.
                    token["lexical_gloss"] = token.get("lexical_gloss") or found.gloss
                elif found.status == "arnold_headword" and looks_like_a_lemma(found.lemma):
                    token["lemma"] = found.lemma
                if found.candidates:
                    # Слово есть в словаре под несколькими лексемами: выбор за
                    # автором (инвариант #7). Документ, сохранённый до
                    # резолвера, иначе остался бы без этого выбора навсегда.
                    # Этот `if` не конфликтует ни с одной веткой выше:
                    # `Resolved` устроен так, что `lexeme_id` и `candidates`
                    # не бывают истинными одновременно (см. `resolve_lexeme`).
                    token["lexeme_candidates"] = list(found.candidates)
                if (token.get("lemma_id"), token.get("lemma"),
                    token.get("lexical_gloss"),
                    tuple(token.get("lexeme_candidates") or ())) != before:
                    touched += 1
        if touched:
            store.save(document)
            by_document.append({"document_id": document["document_id"],
                                 "tokens": touched})
            total += touched
    return {"documents": len(by_document), "tokens": total,
            "by_document": by_document}


def reassign_tokens(store, targets: list[tuple[str, str]], target_lexeme_id: str,
                    lemma: str, gloss: str | None = None) -> dict:
    """Reassign an explicit list of `(document_id, token_id)` tokens to
    `target_lexeme_id` — the scoped sibling of `attach_lexeme`: that one
    rewrites *every* token sharing one old lexeme id, this one rewrites only
    the named tokens, regardless of what lexeme id each currently carries.
    Used when an author has identified specific mis-glossed tokens rather
    than a whole staged lexeme to fold away.
    """
    by_document_ids: dict[str, set[str]] = {}
    for document_id, token_id in targets:
        by_document_ids.setdefault(document_id, set()).add(token_id)
    by_document, total = [], 0
    for document_id, token_ids in by_document_ids.items():
        document = store.load(document_id)
        touched = 0
        for sentence in document.get("sentences") or []:
            for token in sentence.get("tokens") or []:
                if token.get("token_id") not in token_ids:
                    continue
                token["lemma_id"] = target_lexeme_id
                token["lemma"] = lemma
                token.pop("lexeme_candidates", None)
                # Unlike `attach_lexeme` (which only ever moves tokens OFF a
                # staged/merged-away lexeme, whose own gloss was never
                # authoritative), `reassign` exists specifically to correct a
                # mis-resolved token: if the target has no resolvable gloss
                # yet, keeping the PREVIOUS lexeme's gloss text under the new
                # `lemma_id` would silently carry over a now-wrong meaning.
                token["lexical_gloss"] = gloss or None
                touched += 1
        if touched:
            store.save(document)
            by_document.append({"document_id": document_id, "tokens": touched})
            total += touched
    return {"reassigned_into": target_lexeme_id, "tokens": total,
            "by_document": by_document}


def _snapshot(document_id: str, token: dict) -> dict:
    """The pre-edit state of one token, exactly the fields `attach_lexeme`/
    `clear_lexeme_tokens` go on to overwrite — captured before that
    overwrite so a later retraction (see `restore_touched_tokens`) can put
    each named token back exactly as it was. `lexeme_candidates` is
    recorded as `None` when absent (it is always a non-empty list when
    present — see `resolve_lexeme`/`rewrite_minted_lexemes` — so `None`
    unambiguously means "the field was not there")."""
    return {
        "document_id": document_id,
        "token_id": token.get("token_id"),
        "lemma_id": token.get("lemma_id"),
        "lemma": token.get("lemma"),
        "lexical_gloss": token.get("lexical_gloss"),
        "lexeme_candidates": token.get("lexeme_candidates"),
    }


def restore_touched_tokens(store, touched_tokens: list[dict]) -> dict:
    """Exact inverse of `attach_lexeme`/`clear_lexeme_tokens`: put each named
    token back to the pre-edit values recorded in its `_snapshot`. This is
    what makes retracting a merge or delete an actual "undo" of saved
    documents, not just a journal tombstone — see `app.py`'s
    `lexicon_edit_retract`, which calls this with the retracted edit's own
    `touched_tokens`."""
    by_document: dict[str, list[dict]] = {}
    for snapshot in touched_tokens or ():
        by_document.setdefault(snapshot["document_id"], []).append(snapshot)
    restored_by_document, total = [], 0
    for document_id, snapshots in by_document.items():
        document = store.load(document_id)
        by_token_id = {s["token_id"]: s for s in snapshots}
        touched = 0
        for sentence in document.get("sentences") or []:
            for token in sentence.get("tokens") or []:
                snapshot = by_token_id.get(token.get("token_id"))
                if snapshot is None:
                    continue
                token["lemma_id"] = snapshot["lemma_id"]
                token["lemma"] = snapshot["lemma"]
                token["lexical_gloss"] = snapshot["lexical_gloss"]
                if snapshot["lexeme_candidates"] is None:
                    token.pop("lexeme_candidates", None)
                else:
                    token["lexeme_candidates"] = snapshot["lexeme_candidates"]
                touched += 1
        if touched:
            store.save(document)
            restored_by_document.append({"document_id": document_id, "tokens": touched})
            total += touched
    return {"restored_tokens": total, "by_document": restored_by_document}


def clear_lexeme_tokens(store, lexeme_id: str) -> dict:
    """Walk every saved document and un-resolve every token pointing at
    `lexeme_id` — the document-walking half of a lexeme deletion (see
    `app.py`'s `DELETE /api/lexicon/{lexeme_id}`). A deleted lexeme's tokens
    go straight back to the ordinary unresolved re-review queue, per an
    explicit product decision, not into some new "orphaned" state — so this
    clears exactly the fields `build_token` leaves unset for a token the
    resolver has not yet touched (`lemma_id`/`lemma`/`lexical_gloss`, all
    `None`), same shape as `editor/session.py::build_token`.

    The report's `touched_tokens` is each cleared token's PRE-delete
    snapshot (see `_snapshot`) — the caller (`app.py`'s `lexicon_delete`)
    stores this in the journal row so a later retraction can restore it
    exactly (`restore_touched_tokens`).
    """
    by_document, touched_tokens, total = [], [], 0
    for listed in store.list():
        document = store.load(listed["document_id"])
        touched = 0
        for sentence in document.get("sentences") or []:
            for token in sentence.get("tokens") or []:
                if token.get("lemma_id") != lexeme_id:
                    continue
                touched_tokens.append(_snapshot(document["document_id"], token))
                token["lemma_id"] = None
                token["lemma"] = None
                token["lexical_gloss"] = None
                token.pop("lexeme_candidates", None)
                touched += 1
        if touched:
            store.save(document)
            by_document.append({"document_id": document["document_id"],
                                "tokens": touched})
            total += touched
    return {"cleared_lexeme_id": lexeme_id, "tokens": total,
            "by_document": by_document, "touched_tokens": touched_tokens}


def _find_token_in_document(document: dict, token_id: str) -> dict | None:
    for sentence in document.get("sentences") or []:
        for token in sentence.get("tokens") or []:
            if token.get("token_id") == token_id:
                return token
    return None


def resolve_ambiguous_occurrences(store, document_id: str, token_id: str,
                                  alternative_index: int, analyzer) -> dict:
    """Resolve one token's picked `alternatives` entry (`session.
    resolve_ambiguous_token`), then retroactively apply the identical fix to
    every other saved-document token stuck on the exact same word the exact
    same way — an explicit product requirement: picking a reading for one
    occurrence should not leave every other occurrence of the same word
    still asking the author to resolve it again from scratch.

    "The same word stuck the same way" is deliberately narrow: a token only
    qualifies if its `surface` matches (NFC, casefolded) AND it has no
    `lemma_id` yet AND its `alternatives` list is byte-identical to the
    named token's own (before that token is resolved). Identical
    `alternatives` means the same `alternative_index` selects the same
    reading everywhere, so no separate judgement call is needed per token —
    weaker matching (e.g. surface alone) would risk silently resolving a
    token whose competing readings are not actually the same choice.

    Returns a report shaped like `rewrite_minted_lexemes`/`clear_lexeme_
    tokens`: `{"documents": N, "tokens": M, "by_document": [...], "primary":
    <resolve_ambiguous_token's own report>}` — `tokens`/`documents` count
    every touched token/document, the primary one included, so the report's
    top-level numbers describe the whole effect of the action, and
    `"primary"` carries the named token's own outcome for a caller that
    only cares about that one (e.g. what to show inline where the author
    clicked).
    """
    # Imported locally, not at module scope: `session.py` itself imports this
    # module (`from maalula_glosser.editor import lexeme_resolve as LR`) to
    # call `resolve_lexeme`, so a top-level import here would be circular.
    from maalula_glosser.editor import session as S

    document = store.load(document_id)
    token = _find_token_in_document(document, token_id)
    if token is None:
        raise ValueError(f"token {token_id} not found in document {document_id}")

    original_alternatives = token.get("alternatives") or []
    original_surface = nfc(token.get("surface") or "").casefold()

    primary_report = S.resolve_ambiguous_token(token, alternative_index, analyzer)
    store.save(document)

    by_document = [{"document_id": document_id, "tokens": 1}]
    total = 1
    documents_touched = {document_id}

    for listed in store.list():
        other_document_id = listed["document_id"]
        other_document = (document if other_document_id == document_id
                          else store.load(other_document_id))
        touched = 0
        for sentence in other_document.get("sentences") or []:
            for other_token in sentence.get("tokens") or []:
                if other_document_id == document_id and other_token is token:
                    continue
                surface = nfc(other_token.get("surface") or "").casefold()
                if (surface != original_surface
                        or other_token.get("lemma_id") is not None
                        or (other_token.get("alternatives") or []) != original_alternatives):
                    continue
                S.resolve_ambiguous_token(other_token, alternative_index, analyzer)
                touched += 1
        if touched:
            store.save(other_document)
            if other_document_id in documents_touched:
                for entry in by_document:
                    if entry["document_id"] == other_document_id:
                        entry["tokens"] += touched
            else:
                by_document.append({"document_id": other_document_id, "tokens": touched})
                documents_touched.add(other_document_id)
            total += touched

    return {"documents": len(by_document), "tokens": total,
            "by_document": by_document, "primary": primary_report}


def attach_lexeme(store, staged_lexeme_id: str, target_lexeme_id: str,
                  lemma: str, gloss: str | None = None,
                  clear_stale_gloss: bool = False) -> dict:
    """Слить продвигаемое слово с существующей лексемой во всех документах.

    Продвижение снимает ту неоднозначность, ради которой `token` мог нести
    `lexeme_candidates` (Task 6) — список стал бы устаревшим, если бы остался.

    `clear_stale_gloss` defaults to False to preserve promotion's existing
    behavior: a staged token's own already-correct gloss is authoritative
    and worth keeping even when the promoted-into lexeme has no resolvable
    gloss of its own yet. `lexicon_merge` (an arbitrary already-established
    lexeme merged into another) passes True instead — there the source
    lexeme's gloss is not authoritative for the target's meaning, and
    leaving it behind under the target's `lemma_id` would silently carry
    over a now-wrong meaning, the same bug `reassign_tokens` was fixed for.

    The report's `touched_tokens` is each rewritten token's PRE-merge
    snapshot (see `_snapshot`) — `app.py`'s `lexicon_merge` stores this in
    the journal row so a later retraction can restore it exactly
    (`restore_touched_tokens`)."""
    by_document, touched_tokens, total = [], [], 0
    for listed in store.list():
        document = store.load(listed["document_id"])
        touched = 0
        for sentence in document.get("sentences") or []:
            for token in sentence.get("tokens") or []:
                if token.get("lemma_id") != staged_lexeme_id:
                    continue
                touched_tokens.append(_snapshot(document["document_id"], token))
                token["lemma_id"] = target_lexeme_id
                token["lemma"] = lemma
                token.pop("lexeme_candidates", None)
                if gloss:
                    token["lexical_gloss"] = gloss
                elif clear_stale_gloss:
                    token["lexical_gloss"] = None
                touched += 1
        if touched:
            store.save(document)
            by_document.append({"document_id": document["document_id"],
                                "tokens": touched})
            total += touched
    return {"merged_into": target_lexeme_id, "tokens": total,
            "by_document": by_document, "touched_tokens": touched_tokens}
