"""Форма глоссы и форма леммы — одно правило для всех слоёв.

Лексическая глосса — это всё, что есть в токене, кроме морфологии
(`be.aware.SC` → `be.aware`). Лемма — это форма, а не глосса: строка с
граммемой (`car.F`), сегментация (`apt-aṯ`) и арабская вставка леммой не
бывают. Наборы ярлыков перенесены сюда из `morphology/integration.py`, чтобы
существовал один источник правды.
"""
from __future__ import annotations

import re
import unicodedata

from maalula_glosser.editor import arabic_marks

TAM_LABELS = frozenset({"pst", "prs", "prf", "sbjv", "ipv", "sc", "pc", "nc",
                        "nc1", "nc2"})
# "do"/"io" сознательно исключены: у šwy лексическая глосса — глагол "do".
# hes/inch/ins/pn/pos/pp/prsv/recp/rel — ярлыки из реестра проекта
# (data/grammar_kb/glossing_policy.json), а не догадка.
STRUCT_LABELS = frozenset({"intf", "neg", "hd", "free", "cns", "indf", "def",
                           "exist", "dom", "num", "aff", "emp", "hes", "inch",
                           "ins", "pn", "pos", "pp", "prsv", "recp", "rel"})
# Третья альтернатива — голое лицо+род без числа (`3M`, `F`): встречается в
# том же реестре как отдельный ярлык, не только в связке с s/p.
PNG_RE = re.compile(r"^[123]?(m|f|c)?(s|p)$|^(mp|fp|ms|fs)$|^[123]?[mfc]$",
                    re.I)

# Точка, за которой начинается грамматический хвост. Точка соединяет и
# лексические слова: `fell.out.PST` — это не `fell.PST`.
_GRAMMATICAL_TAIL = re.compile(r"\.(?=[A-Z0-9?*]+(?:[.=+\-]|$))")
_SEPARATORS = re.compile(r"[.=+\-]")


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value or "")


def lexical_head(gloss: str) -> str:
    return _GRAMMATICAL_TAIL.split(gloss or "", maxsplit=1)[0]


def is_grammatical_label(part: str) -> bool:
    part = (part or "").strip()
    if not part:
        return False
    lowered = part.lower()
    return (lowered in TAM_LABELS or lowered in STRUCT_LABELS
            or bool(PNG_RE.match(part)) or part.isdigit())


def has_lexical_part(gloss: str) -> bool:
    """Есть ли в глоссе хоть одна не-грамматическая часть.

    Регистр не используется как критерий: в авторской IGT граммемы пишутся
    строчными (`3fs`), и регистровая эвристика принимала их за значение.
    """
    parts = [p for p in _SEPARATORS.split(nfc(gloss)) if p.strip()]
    return any(not is_grammatical_label(part) for part in parts)


def could_be_a_typed_meaning(gloss: str) -> bool:
    """Автор ввёл это в поле значения руками.

    Морфологии здесь нет, поэтому слово, лишь совпавшее с именем структурного
    ярлыка (`free`, `exist`), — это значение. Ярлык, записанный так, как ярлыки
    и пишутся (`FREE`), и признаки лица/числа в любом регистре (`3fs`, `mp`)
    значением не становятся.
    """
    text = nfc(gloss).strip()
    if has_lexical_part(text):
        return True
    parts = [part for part in _SEPARATORS.split(text) if part.strip()]
    return (len(parts) == 1 and parts[0].lower() in STRUCT_LABELS
            and not parts[0].isupper())


def looks_like_a_lemma(value: str) -> bool:
    value = nfc(value).strip()
    if not value:
        return False
    if lexical_head(value) != value:
        return False
    if "-" in value or "=" in value or "+" in value:
        return False
    return not arabic_marks.spans(value)
