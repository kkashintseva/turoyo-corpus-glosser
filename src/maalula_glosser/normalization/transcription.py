from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


# Two subscript marks record an allophone, not a phoneme: the ring below marks
# a devoiced b̥/d̥ (ġurəb̥ṯa, Aid̥faʕ, rab̥) and the tilde below a further b̰.
# They occur only in transcriptions of running text — not one lexicon entry,
# paradigm cell or clitic is spelled with them — so on the lookup side they can
# only ever prevent a match. Folded away before anything is looked up; the
# transcription the author typed keeps them.
ALLOPHONE_MARKS = "̥̰"  # COMBINING RING BELOW, COMBINING TILDE BELOW
_ALLOPHONES = {ord(mark): None for mark in ALLOPHONE_MARKS}


def fold_allophones(value: str) -> str:
    """Drop the allophonic subscript marks, keeping every distinctive one.

    Decomposes first, so a precomposed letter that *contains* one of the marks
    is folded too, and the dot below of the emphatics — a different mark, and a
    contrastive one — is never touched.
    """
    decomposed = unicodedata.normalize("NFD", value or "")
    return unicodedata.normalize("NFC", decomposed.translate(_ALLOPHONES))


@dataclass(frozen=True)
class NormalizationStep:
    source: str
    target: str
    position: int
    status: str


@dataclass(frozen=True)
class NormalizationResult:
    original: str
    nfc: str
    canonical: str
    steps: tuple[NormalizationStep, ...]
    confidence: float


class TranscriptionNormalizer:
    """Unicode-aware simultaneous transcription mapping.

    The confirmed mapping ḳ→k and k→c must be simultaneous. This class uses
    longest-match scanning over the original string, never chained replace().
    """

    def __init__(self, mappings: Iterable[dict]):
        confirmed = [item for item in mappings if item.get("status") == "confirmed"]
        self.mapping = {str(item["source"]): str(item["target"]) for item in confirmed}
        self.status = {str(item["source"]): str(item.get("status", "confirmed")) for item in confirmed}
        self.keys = sorted(self.mapping, key=len, reverse=True)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TranscriptionNormalizer":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(data.get("mappings", []))

    def normalize(self, value: str, source_system: str = "arnold") -> NormalizationResult:
        original = value
        nfc = unicodedata.normalize("NFC", value)
        if source_system in {"canonical", "modern", "glossing_updated", "masc"}:
            return NormalizationResult(original, nfc, nfc, tuple(), 1.0)
        if source_system != "arnold":
            raise ValueError(f"Unknown transcription source system: {source_system}")
        out: list[str] = []
        steps: list[NormalizationStep] = []
        i = 0
        while i < len(nfc):
            matched = False
            for source in self.keys:
                if nfc.startswith(source, i):
                    target = self.mapping[source]
                    out.append(target)
                    if source != target:
                        steps.append(NormalizationStep(source, target, i, self.status[source]))
                    i += len(source)
                    matched = True
                    break
            if not matched:
                out.append(nfc[i])
                i += 1
        canonical = unicodedata.normalize("NFC", "".join(out))
        confidence = 1.0 if all(step.status == "confirmed" for step in steps) else 0.75
        return NormalizationResult(original, nfc, canonical, tuple(steps), confidence)

    def canonical(self, value: str, source_system: str = "arnold") -> str:
        return self.normalize(value, source_system=source_system).canonical
