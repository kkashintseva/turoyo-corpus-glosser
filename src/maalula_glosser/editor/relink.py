"""«Перепривязать леммы»: пройти сохранённый документ и привязать к словнику
то, что автор уже проверил, но что осталось без лексемы или привязано не туда.

Трогаются только токены, которые правил человек (`locked` или хоть одна
ячейка с `origin == user`): морфология и глоссы не меняются никогда, меняются
лишь `lemma_id` / `lemma` / `lexical_gloss` / `lexeme_candidates`.

Порядок для одного токена:
1. лексема слита (lexicon_edits merge) -> цель слияния;
2. лексемы нет или она минтованная (`lexnew_`) -> `resolve_lexeme`;
3. глагольная глосса (NC/NC1/NC2/PC/SC…) на не-глаголе -> глагол того же
   корня (D5; корень берётся из лексемы и из согласных основы, D3);
   выбор среди глаголов корня: порода по форме основы -> совпадение значения
   -> разбор анализатора -> единственный глагол; если ни один не выбран,
   автору остаются `lexeme_candidates`;
4. глагола корня в словнике проекта нет, а у Арнольда он есть (isbaḳ yisbuḳ
   для sabeklun) -> новая лексема `lexnew_` со словарной формой Арнольда;
5. ничего не нашлось -> `lexnew_` по основе, как при импорте
   (`session.overlay_user_igt`): слово попадает в очередь Словника, откуда
   его можно отправить «в словарик».

Имена (`PN`, глосса с заглавной), числа и арабские вставки (ᴬ…ᴬ) лексем не
получают.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from maalula_glosser.arnold_verbs import arnold_verbs
from maalula_glosser.editor import arabic_marks
from maalula_glosser.editor import lexeme_resolve as LR
from maalula_glosser.editor import session as S
from maalula_glosser.gloss_shape import lexical_head, looks_like_a_lemma
from maalula_glosser.io_utils import stable_id
from maalula_glosser.normalization.transcription import fold_allophones
from maalula_glosser.pos import citation_class, gloss_is_verbal, stem_classes, verbal_trigger

VOWELS = set("aeiouāēīōūəAEIOUĀĒĪŌŪ")
_NAME_OR_NUMBER = re.compile(r"PN|\d{2,}$")


def nfc(value: str | None) -> str:
    return unicodedata.normalize("NFC", value or "")


# ---------------------------------------------------------------------------
# the token's lexical cell
# ---------------------------------------------------------------------------
def host_stem(cells: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    """`session._lexical_stem`, plus: a proclitic (a cell right before a `=`
    cell — b=, maʕl=, l=) is not the host, so `b=blōt-a` gives `blōt`; and a
    name or a number as the host gives nothing (`b=Maʕlūla` is not `b-`)."""
    if any(c.get("sep") in {"=", "+"} and _NAME_OR_NUMBER.match(c.get("gloss") or "")
           for c in cells):
        return None, None
    stem, head = S._lexical_stem(cells)
    if stem:
        return stem, head
    lexical = [i for i, c in enumerate(cells) if S._looks_lexical_gloss(c.get("gloss") or "")]
    hosts = [i for i in lexical
             if not (i + 1 < len(cells) and cells[i + 1].get("sep") in {"=", "+"})]
    if len(lexical) > 1 and len(hosts) == 1:
        cell = cells[hosts[0]]
        return nfc(cell.get("form")) or None, lexical_head(cell.get("gloss") or "") or None
    return None, None


def host_word(cells: list[dict[str, Any]], stem: str | None) -> str | None:
    """The host with its own suffixes but without clitics: `ʕa=ṣlīb-a` ->
    `ṣlība`, `fark-a` -> `farka`. Free-state nouns are listed with their -a,
    so the bare stem alone (`ṣlīb`) finds no attested form."""
    if not stem:
        return None
    for index, cell in enumerate(cells):
        if nfc(cell.get("form")) != stem:
            continue
        word = stem
        for after in cells[index + 1:]:
            if after.get("sep") in {"=", "+"}:
                break
            word += nfc(after.get("form"))
        return word
    return None


def _mintable(stem: str) -> bool:
    """A stem worth a queue row of its own: not a one-letter clitic (u, e),
    not an Arabic insert, not a segmentation that slipped into the stem."""
    return (len(stem) > 1 and not arabic_marks.opens(stem) and not arabic_marks.closes(stem)
            and not re.search(r"[-=.<>\s]", stem) and looks_like_a_lemma(stem))


def _is_name(head: str | None, gloss: str) -> bool:
    return bool(head and head[:1].isupper()) or "PN" in re.split(r"[-=.]", gloss)


def _consonants(stem: str) -> list[str]:
    return [ch for ch in nfc(stem) if (ch.isalpha() or ch in "ʔʕ") and ch not in VOWELS]


def roots_for_stem(analyzer, stem: str, known: set[str] | None = None) -> list[tuple[str, str]]:
    """Candidate roots for a stem (D3 and a few regular alternations), each
    tagged with where it came from: 'skeleton' (the consonants as they stand),
    'prefix' (a derived-stem č-/n-/m-/t- or a Gt infix dropped), 'fill' (a
    biconsonantal remnant completed by a guessed radical). Only roots some
    lexeme — or `known` — has are returned."""
    cs = _consonants(fold_allophones(stem or ""))
    raw = "".join(cs)
    collapsed = "".join(c for i, c in enumerate(cs) if i == 0 or c != cs[i - 1])
    out: list[tuple[str, str]] = []
    for base in (collapsed, raw):
        out.append((base, "skeleton"))
        if base[:1] in ("č", "n", "m", "t") and len(base) > 3:
            out.append((base[1:], "prefix"))
        if len(base) > 3 and base[1] == "č" and base[0] != "m":
            out.append((base[0] + base[2:], "prefix"))       # Gt: ščaġl -> šġl
        if base[:2] in ("mč", "mn") and len(base) > 4:
            out.append((base[2:], "prefix"))
        if len(base) > 3:
            out.append((base[-3:], "prefix"))
    for base in {raw, raw[1:] if raw[:1] in ("m", "č", "n") else raw}:
        if len(base) == 3 and base[1] == base[2]:
            out.append((base[:2] + "y", "fill"))                 # mʕann -> ʕny
    for base in (collapsed, raw):
        if len(base) == 3 and base[0] in ("m", "č", "n", "t"):
            base = base[1:]
        if len(base) == 2:
            out += [(r, "fill") for r in (base + base[-1], "n" + base, "ʔ" + base,
                                          base[0] + "w" + base[1], base[0] + "y" + base[1],
                                          base + "y")]
    for base in (collapsed,):
        if len(base) > 3:
            out += [(base[i:i + 3], "fill") for i in range(1, len(base) - 2)]
    known = known or set()
    seen, result = set(), []
    for root, kind in out:
        if root and root not in seen and (root in analyzer.lexeme_ids_by_root or root in known):
            seen.add(root)
            result.append((root, kind))
    return result


# ---------------------------------------------------------------------------
# the pass
# ---------------------------------------------------------------------------
class Relinker:
    def __init__(self, analyzer, merges: dict[str, str] | None = None):
        self.analyzer = analyzer
        self.merges = dict(merges or {})
        is_flex = getattr(analyzer, "profile", None) == "flex"
        inventory_name = "gloss_inventory.flex.json" if is_flex else "gloss_inventory.json"
        inventory = Path(analyzer.root) / "config" / inventory_name
        display = {}
        if inventory.exists():
            display = json.loads(inventory.read_text(encoding="utf-8")).get("display_map") or {}
        self.trigger = verbal_trigger(display)
        # Never consult the inherited Maalula/Arnold lexicon in the Turoyo
        # profile, even if an old data/ directory is retained beside the code.
        self.arnold = {} if is_flex else arnold_verbs(analyzer)
        self._votes: dict[str, set[str]] = {}

    # -- helpers -----------------------------------------------------------
    def final(self, lexeme_id: str | None) -> str | None:
        seen = set()
        while lexeme_id in self.merges and lexeme_id not in seen:
            seen.add(lexeme_id)
            lexeme_id = self.merges[lexeme_id]
        return lexeme_id

    def lemma(self, lexeme_id: str | None) -> str | None:
        return self.analyzer.lexeme_record(lexeme_id).get("lemma_canonical")

    def resolve(self, surface: str, stem: str, whole: str | None, head: str | None):
        """`resolve_lexeme`, host word first. The bare stem of a noun with
        endings is only trusted when the meaning does not contradict it:
        `lōġ` of `lōġ-ṯ-a` 'language' is a MASC form of hōḏ(i) 'this'."""
        bare = re.sub(r"^[^\w]+|[^\w]+$", "", surface.replace("‿", ""))
        tries = []
        if whole and whole != stem:
            tries.append((whole, False))
        tries.append((stem, bool(whole and whole != stem)))
        fallback = None
        for key, guarded in tries:
            found = LR.resolve_lexeme(self.analyzer, fold_allophones(bare), fold_allophones(key))
            if not found:
                continue
            if found.lexeme_id:
                if (guarded and found.basis != "attested form"
                        and self.agrees(head, self.final(found.lexeme_id)) is False):
                    continue
                return found
            if found.candidates:
                return found
            fallback = fallback or found
        return fallback

    def meaning(self, lexeme_id: str) -> str | None:
        gloss, _status = self.analyzer.lexical.resolve(lexeme_id)
        return gloss

    @staticmethod
    def _words(text: str | None) -> set[str]:
        return {w for w in re.split(r"[\s.,;/()]+", (text or "").lower())
                if w and w not in {"be", "to", "a", "the", "of"}}

    def agrees(self, head: str | None, lexeme_id: str) -> bool | None:
        mine, theirs = self._words(head), self._words(self.meaning(lexeme_id))
        if not mine or not theirs:
            return None
        return mine <= theirs or theirs <= mine

    def votes(self, word: str) -> set[str]:
        if word not in self._votes:
            found: set[str] = set()
            try:
                for produced in self.analyzer.all_stage_candidates(word)["produced"]:
                    lexeme_id = produced["candidate"].lemma_id
                    if lexeme_id:
                        found.add(self.final(lexeme_id))
            except Exception:  # a diagnostic, never a reason to fail the pass
                pass
            self._votes[word] = found
        return self._votes[word]

    def pick_verb(self, verbs: list[str], root: str, kind: str, stem: str,
                  head: str | None, gloss: str, word: str) -> tuple[str | None, str | None]:
        if not verbs:
            return None, None
        shape = stem_classes(stem, root, gloss)
        if shape:
            fitting = [v for v in verbs if citation_class(self.lemma(v) or "", root) in shape]
            if len(fitting) == 1 and (kind != "fill" or len(verbs) > 1):
                return fitting[0], "порода по форме основы"
            if fitting:
                verbs = fitting
        agreeing = [v for v in verbs if self.agrees(head, v)]
        if len(agreeing) == 1:
            return agreeing[0], "значение совпадает"
        pool = agreeing or verbs
        voted = [v for v in pool if v in self.votes(word)] if word else []
        if len(voted) == 1 and (agreeing or kind != "fill"):
            return voted[0], "форму разбирает анализатор"
        if len(verbs) == 1 and kind != "fill":
            return verbs[0], "единственный глагол корня"
        return None, None

    def pick_arnold(self, roots: list[tuple[str, str]], stem: str,
                    gloss: str) -> tuple[dict | None, list[dict]]:
        seen: list[dict] = []
        for root, kind in roots:
            entries = self.arnold.get(root) or []
            if not entries:
                continue
            shape = stem_classes(stem, root, gloss)
            fitting = [e for e in entries
                       if not shape or citation_class(e["citation"], root) in shape] or entries
            if len(fitting) == 1 and kind != "fill":
                return {**fitting[0], "root": root}, []
            if kind != "fill":
                seen += [{**e, "root": root} for e in fitting]
        return None, seen

    # -- one token -----------------------------------------------------------
    def token(self, token: dict[str, Any], arabic: bool) -> dict[str, Any] | None:
        cells = token.get("morphemes") or []
        if not cells or S.is_speaker(token):
            return None
        if not (token.get("locked") or any(c.get("origin") == S.USER for c in cells)):
            return None
        gloss = "".join((c.get("sep") or "") + (c.get("gloss") or "") for c in cells)
        stem, head = host_stem(cells)
        surface = nfc(token.get("surface"))
        word = surface.split()[-1] if surface else ""
        word = re.sub(r"^[^\w]+|[^\w]+$", "", fold_allophones(word).replace("‿", ""))
        word = word[:1].lower() + word[1:]
        current = token.get("lemma_id")
        new, why, candidates, lemma_text = current, None, [], None
        is_verbal = gloss_is_verbal(gloss, self.trigger)
        citation = None if is_verbal else S.nominal_citation(cells, stem)

        if current in self.merges:
            new, why = self.final(current), "лексема слита"
        elif (not current or current.startswith(("lexnew_", "lexsvc_"))) and stem and not arabic:
            found = self.resolve(surface, stem, host_word(cells, stem), head)
            if found and found.lexeme_id:
                new, why = self.final(found.lexeme_id), "нашлась в словнике: " + found.basis
            elif found and found.candidates:
                candidates = [self.final(c) for c in found.candidates]
                why = "в словнике несколько лексем с этой формой"
            elif found and found.status == "arnold_headword" and looks_like_a_lemma(found.lemma):
                lemma_text = found.lemma
        if new and (self.lemma(new) or "").startswith("["):
            new, why = None, "скобочная псевдолексема"

        real = bool(new) and not new.startswith("lexnew_")
        if is_verbal and stem and not arabic and not (real and self.analyzer.is_verbal(new)):
            own_root = nfc(self.analyzer.lexeme_record(new).get("root_canonical")) if real else ""
            roots = ([(own_root, "lexeme")] if own_root else []) + [
                r for r in roots_for_stem(self.analyzer, stem, set(self.arnold)) if r[0] != own_root]
            # r assimilated to a following l-clitic: amel=lun 'he told them'
            # is amar yīmar, not a root with l
            index = next((i for i, c in enumerate(cells) if nfc(c.get("form")) == stem), None)
            if (stem.endswith("l") and index is not None and index + 1 < len(cells)
                    and nfc(cells[index + 1].get("form")).startswith("l")):
                seen_roots = {r for r, _ in roots}
                roots += [r for r in roots_for_stem(self.analyzer, stem[:-1] + "r", set(self.arnold))
                          if r[0] not in seen_roots]
            verb_candidates: list[str] = []
            for root, kind in roots:
                verbs = sorted(i for i in self.analyzer.lexeme_ids_by_root.get(root, ())
                               if self.analyzer.is_verbal(i))
                chosen, how = self.pick_verb(verbs, root, kind, stem, head, gloss, word)
                if chosen:
                    new, why, verb_candidates, lemma_text = chosen, f"глагол √{root}: {how}", [], None
                    break
                verb_candidates += verbs
            else:
                if verb_candidates:
                    candidates = list(dict.fromkeys(verb_candidates))
                    why = "глагольная глосса: выберите глагол"
                    if real:
                        new = None
                else:
                    # No verb of the root in the project lexicon: the same
                    # service verb the analyzer binds (`lexsvc_<root>`), under
                    # Arnold's citation when the stem's shape picks one entry.
                    entry, others = self.pick_arnold(roots, stem, gloss)
                    if entry or others:
                        root = (entry or others[0])["root"]
                        new = self.analyzer.service_verb(root)
                        lemma_text = entry["citation"] if entry else None
                        why = (f"глагола √{root} нет в словнике проекта; у Арнольда "
                               + (f"{entry['citation']} (с. {entry['page']})" if entry else
                                  ", ".join(e["citation"] for e in others[:4])))

        if (new is None and not candidates and not arabic and stem
                and not _is_name(head, gloss)):
            plain = nfc(fold_allophones(stem))
            if lemma_text or _mintable(plain):
                # the same id the import mints (`overlay_user_igt`): from the stem
                new = stable_id("lexnew", stem.casefold())
                why = why or "нет в словнике: новая лексема (в очередь Словника)"
                lemma_text = lemma_text or citation or plain
        elif (new == current and current and current.startswith("lexnew_") and citation
                and nfc(token.get("lemma")) in (stem, nfc(fold_allophones(stem)))
                and citation != nfc(token.get("lemma"))):
            lemma_text, why = citation, "словарная форма: основа с окончанием"

        before = {k: token.get(k) for k in ("lemma_id", "lemma", "lexical_gloss", "lexeme_candidates")}
        after = dict(before)
        if new != current or lemma_text:
            after["lemma_id"] = new
            after["lemma"] = (lemma_text if new and new.startswith(("lexnew_", "lexsvc_")) and lemma_text
                              else self.lemma(new) if new else None)
        if new != current:
            if new:
                after["lexical_gloss"] = head or token.get("lexical_gloss") or (
                    self.meaning(new) if not new.startswith("lexnew_") else None)
        if candidates:
            after["lexeme_candidates"] = candidates
        if after == before and not (candidates and not before["lemma_id"]):
            return None     # still open choices stay on the list until picked
        if not before["lemma_id"]:
            action = "new" if after["lemma_id"] and after["lemma_id"].startswith(("lexnew_", "lexsvc_")) else (
                "bound" if after["lemma_id"] else "choose")
        elif not after["lemma_id"]:
            action = "unbound"
        elif after["lemma_id"] != before["lemma_id"]:
            action = "new" if after["lemma_id"].startswith(("lexnew_", "lexsvc_")) else "rebound"
        elif candidates:
            action = "choose"
        else:
            action = "renamed"
        return {"token_id": token.get("token_id"), "surface": surface, "gloss": gloss,
                "stem": stem, "head": head, "action": action, "why": why, "before": before, "after": after,
                "candidates": [{"lexeme_id": c, "lemma": self.lemma(c)} for c in candidates]}

    # -- a document -----------------------------------------------------------
    def plan(self, document: dict[str, Any]) -> list[dict[str, Any]]:
        changes = []
        for index, sentence in enumerate(document.get("sentences") or []):
            tokens = sentence.get("tokens") or []
            arabic = S._arabic_insert_token_ids(tokens)
            for token in tokens:
                change = self.token(token, token.get("token_id") in arabic
                                    or arabic_marks.opens(nfc(token.get("surface"))))
                if change:
                    change["line"] = index + 1
                    change["sentence_id"] = sentence.get("sentence_id")
                    changes.append(change)
        return changes


def choose(relinker: "Relinker", change: dict[str, Any], lexeme_id: str) -> None:
    """The author picked one of the candidates in the preview."""
    change["after"] = {**change["after"], "lemma_id": lexeme_id,
                       "lemma": relinker.lemma(lexeme_id),
                       "lexeme_candidates": None}
    head = host_stem_head = None
    if change.get("head"):
        head = change["head"]
    change["after"]["lexical_gloss"] = head or relinker.meaning(lexeme_id)
    change["action"] = "rebound" if change["before"].get("lemma_id") else "bound"
    change["why"] = "выбрано автором"


ACTION_LABELS = {
    "bound": "привязано к словнику",
    "rebound": "перепривязано",
    "new": "новая лексема (очередь Словника)",
    "unbound": "отвязано — выберите лексему",
    "choose": "нужен выбор",
    "renamed": "новая лексема: словарная форма поправлена",
}


def summarize(changes: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for change in changes:
        counts[change["action"]] = counts.get(change["action"], 0) + 1
    return counts


def apply(document: dict[str, Any], changes: list[dict[str, Any]]) -> tuple[list, list]:
    """Write the planned fields into the document's tokens. Returns deep
    copies of the touched tokens before and after, for one undo entry."""
    import copy
    wanted = {change["token_id"]: change["after"] for change in changes}
    before, after = [], []
    for sentence in document.get("sentences") or []:
        for token in sentence.get("tokens") or []:
            fields = wanted.get(token.get("token_id"))
            if fields is None:
                continue
            before.append(copy.deepcopy(token))
            for key, value in fields.items():
                if value is None and key == "lexeme_candidates":
                    token.pop(key, None)
                else:
                    token[key] = value
            after.append(copy.deepcopy(token))
    return before, after


def merges_of(root) -> dict[str, str]:
    from maalula_glosser.lexicon_edits import active_edits
    edits = active_edits(Path(root))
    return {source: row["target_lexeme_id"] for source, row in edits["merges"].items()}
