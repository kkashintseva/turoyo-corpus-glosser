"""Links from a lexical gloss to the Arnold dictionary scan.

The lexical gloss shown in line 3 comes, for most words, from an Arnold entry
that `data/lexicon_v3/lexicon.jsonl` already records a page number for. Where
that binding exists, the editor can offer the page of the scan it came from, so
a doubtful sense can be checked against the source instead of guessed at.

Offline note: building and serving links needs no network. Only *following* a
link does, which is the user's choice at click time.
"""
from __future__ import annotations

import json
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

# Arnold, Das Neuwestaramäische, on the aramaicsite library viewer.
BOOK_ID = "1ded7ea7-c5eb-57cb-a41a-ada59cd9321b"
URL_TEMPLATE = "https://evb0110.github.io/aramaicsite/library/book/{book}?page={page}"

# The viewer's ?page= is assumed to be the printed page number our data records.
# If the scan turns out to be offset (front matter counted, volumes split), set
# `page_offset` in config/arnold_scan.json rather than editing this file.
CONFIG = "config/arnold_scan.json"


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def head(lemma: str) -> str:
    """Head form of a multi-part lemma: `xaṯṯeb yxaṯṯeb` → `xaṯṯeb`.

    Mirrors `morphology.lexical_gloss._head`, so a link is looked up under the
    same key the gloss itself was resolved under.
    """
    text = nfc(lemma).strip()
    return text.split()[0] if text else ""


def candidate_keys(lemma: str | None) -> list[str]:
    """Every form of a project lemma the dictionary might be indexed under.

    Project lemmas cite principal parts two ways: whitespace for a second
    principal part (`akreṣ yakreṣ`) and a slash for variants of one
    (`bess/bessi`, `ikər/ikri yikər/yikri`). Arnold heads the entry with a
    single form, so both have to be peeled off before a lookup can match.
    """
    text = nfc(lemma or "").strip()
    if not text:
        return []
    keys = [text, head(text)]
    keys.extend(part for part in head(text).split("/") if part)
    seen, ordered = set(), []
    for key in keys:
        if key and key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


class ArnoldLinks:
    """lemma → page in the Arnold scan.

    Indexed under every key the lexical gloss layer might have matched on:
    the Arnold `lemma`, its head form, and the canonical headword.
    """

    def __init__(self, root: Path | str = "."):
        self.root = Path(root)
        self.by_lemma: dict[str, int] = {}
        self.book_id = BOOK_ID
        self.page_offset = 0

        settings = self.root / CONFIG
        if settings.exists():
            data = json.loads(settings.read_text(encoding="utf-8"))
            self.book_id = data.get("book_id", self.book_id)
            self.page_offset = int(data.get("page_offset", 0))

        source = self.root / "data/lexicon_v3/lexicon.jsonl"
        if not source.exists():
            return
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            page = entry.get("page")
            if not isinstance(page, int) or page <= 0:
                continue
            for key in (entry.get("lemma"), entry.get("headword"),
                        entry.get("headword_canonical")):
                if key:
                    # setdefault: the first entry for a headword wins, matching
                    # how lexical_gloss resolves a sense for it
                    self.by_lemma.setdefault(nfc(key), page)
                    self.by_lemma.setdefault(head(key), page)

        # Some Arnold entries were recovered well enough to retain their root
        # and page, but not well enough for lexicon_v3 to extract a headword.
        # A project lexeme already bound to the same *unique* root can still be
        # linked safely.  This covers e.g. MASC ``asab yīsub`` -> Arnold
        # ``ʔsb/ʔsp`` p. 37; ambiguous roots deliberately receive no fallback.
        roots_path = (self.root / "data" / "dictionary_morphology_v1"
                      / "roots_and_classes.jsonl")
        lexemes_path = self.root / "data" / "final" / "lexemes.jsonl"
        if roots_path.exists() and lexemes_path.exists():
            pages_by_root: dict[str, set[int]] = defaultdict(set)
            for line in roots_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                root = nfc(entry.get("root") or "")
                page = entry.get("page")
                if root and isinstance(page, int) and page > 0:
                    pages_by_root[root].add(page)
            unique_page = {root: next(iter(pages))
                           for root, pages in pages_by_root.items()
                           if len(pages) == 1}
            for line in lexemes_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                lexeme = json.loads(line)
                lemma = nfc(lexeme.get("lemma_canonical") or "")
                page = unique_page.get(nfc(lexeme.get("root_canonical") or ""))
                if lemma and page:
                    self.by_lemma.setdefault(lemma, page)

    def url(self, page: int) -> str:
        return URL_TEMPLATE.format(book=self.book_id, page=page + self.page_offset)

    def page_for(self, lemma: str | None) -> int | None:
        for key in candidate_keys(lemma):
            page = self.by_lemma.get(key)
            if page:
                return page
        return None

    def link_for(self, lemma: str | None,
                 surface: str | None = None) -> dict[str, Any] | None:
        """`{page, url, match}` for a gloss with dictionary provenance, else None.

        `match` says how the entry was reached, and the tooltip must repeat the
        distinction rather than flatten it:

        - `lemma` — the analysis is bound to a lexeme that heads a dictionary
          entry. This is provenance.
        - `surface` — nothing bound, but the written form itself heads an entry.
          That is a lookup a reader could do by hand, not evidence about this
          token; a homograph can land on the wrong entry.

        Trusted-corpus analyses frequently carry no project lemma at all, so
        without the second tier most A1 tokens would offer no link.
        """
        page = self.page_for(lemma)
        if page is not None:
            return {"page": page, "url": self.url(page), "match": "lemma"}
        page = self.page_for(surface)
        if page is not None:
            return {"page": page, "url": self.url(page), "match": "surface"}
        return None

    def annotate(self, document: dict[str, Any]) -> dict[str, Any]:
        """Attach `dictionary` to every token, None where there is no entry."""
        for sentence in document.get("sentences") or []:
            for token in sentence.get("tokens") or []:
                token["dictionary"] = self.link_for(
                    token.get("lemma"), token.get("normalized"))
        return document
