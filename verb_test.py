#!/usr/bin/env python3
"""Standalone Turoyo verb-parser tester.

Usage:

    # Web tester:
    python verb_test.py

    # Terminal tester:
    python verb_test.py mzabhile kəmʕawaḷan

This tool does NOT read or modify editor tokens.  It calls the rule engine
directly, so it is useful for separating parser-rule problems from UI/token
eligibility problems.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys
import threading
import webbrowser
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

from maalula_glosser.flex_corpus import FlexCorpusTextGlosser
from maalula_glosser.turoyo_verbs import (
    BOUNDARIES,
    CORE_RULES,
    PREVERBS,
    TuroyoVerbParser,
    consonant_skeleton,
)


def find_root() -> Path:
    here = Path.cwd().resolve()
    root = next(
        (candidate for candidate in (here, *here.parents)
         if (candidate / "pyproject.toml").is_file()),
        None,
    )
    if root is None:
        raise SystemExit(
            "Не найден корень проекта (pyproject.toml). "
            "Запусти verb_test.py из папки turoyo-corpus-glosser."
        )
    return root


def make_parser(root: Path) -> TuroyoVerbParser:
    """Load the same FLEx corpus analyzer used by the editor."""
    glosser = FlexCorpusTextGlosser(root)
    return TuroyoVerbParser(glosser.analyzer)


def diagnose(parser: TuroyoVerbParser, surface: str, *, limit: int = 30) -> dict[str, Any]:
    surface = str(surface or "").strip()
    suggestions = parser.suggest(surface, limit=limit)

    suffix_matches: list[dict[str, Any]] = []
    folded = surface.casefold()

    for rule in CORE_RULES:
        if len(surface) <= len(rule.suffix):
            continue
        if not folded.endswith(rule.suffix.casefold()):
            continue

        stem = surface[:-len(rule.suffix)]
        evidence = parser.evidence.matches(stem, limit=5)
        suffix_matches.append({
            "rule": rule.name,
            "label": rule.label,
            "family": rule.family,
            "suffix": rule.suffix,
            "stem": stem,
            "skeleton": consonant_skeleton(stem),
            "evidence": [
                {
                    "stem": row.stem,
                    "gloss": row.gloss,
                    "match": kind,
                    "count": row.count,
                }
                for row, kind, _bonus in evidence
            ],
            "emittable_without_evidence": len(consonant_skeleton(stem)) >= 2,
        })

    preverb_matches: list[dict[str, str]] = []
    for prefix, gloss in PREVERBS:
        if len(surface) > len(prefix) and folded.startswith(prefix.casefold()):
            preverb_matches.append({
                "prefix": prefix,
                "gloss": gloss,
                "host": surface[len(prefix):],
            })

    if any(mark in surface for mark in BOUNDARIES):
        reason = (
            "В форме уже есть явная граница -, = или +. "
            "Основной parser такие формы намеренно не переразбирает."
        )
    elif suggestions:
        reason = f"Найдено разборов: {len(suggestions)}."
    elif suffix_matches:
        too_short = all(
            not row["emittable_without_evidence"] and not row["evidence"]
            for row in suffix_matches
        )
        if too_short:
            reason = (
                "Окончания совпали, но после их снятия остаётся слишком короткий "
                "согласный каркас и нет подтверждения во FLEx."
            )
        else:
            reason = (
                "Окончания совпали, но итоговый список предложений пуст. "
                "Смотри блок «Совпавшие правила» — это уже ошибка/ограничение parser'а."
            )
    elif preverb_matches:
        reason = (
            "Похожий преверб найден, но его host не получил ни одного "
            "глагольного разбора."
        )
    else:
        reason = "Ни одно из текущих правил глагольного окончания не совпало."

    return {
        "surface": surface,
        "skeleton": consonant_skeleton(surface),
        "reason": reason,
        "suggestions": [row.as_dict() for row in suggestions],
        "suffix_matches": suffix_matches,
        "preverb_matches": preverb_matches,
    }


def print_result(result: dict[str, Any]) -> None:
    print()
    print("=" * 72)
    print(result["surface"], f"  [C: {result['skeleton'] or '∅'}]")
    print(result["reason"])

    if result["suggestions"]:
        print("\nПРЕДЛОЖЕНИЯ:")
        for i, row in enumerate(result["suggestions"], start=1):
            evidence = row.get("evidence")
            if evidence:
                ev = (
                    f"{evidence.get('match')}: "
                    f"{evidence.get('stem') or '—'} / "
                    f"{evidence.get('gloss') or '—'}"
                )
            else:
                ev = "без FLEx evidence"
            print(f"  {i}. {row['segmentation']}")
            print(f"     {row['gloss']}")
            print(
                f"     {row.get('confidence', '')} · "
                f"{row.get('rule_label') or row.get('rule')} · {ev}"
            )

    if result["suffix_matches"]:
        print("\nСОВПАВШИЕ ПРАВИЛА:")
        for row in result["suffix_matches"]:
            ev = row["evidence"]
            ev_text = (
                "; ".join(
                    f"{e['stem']}/{e['gloss']} ({e['match']})"
                    for e in ev
                )
                if ev else "нет"
            )
            print(
                f"  - {row['suffix']} → {row['label']}; "
                f"stem={row['stem']}; C={row['skeleton'] or '∅'}; "
                f"FLEx={ev_text}"
            )

    if result["preverb_matches"]:
        print("\nВОЗМОЖНЫЕ ПРЕВЕРБЫ:")
        for row in result["preverb_matches"]:
            print(
                f"  - {row['prefix']}= ({row['gloss']}); host={row['host']}"
            )


PAGE = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Тест глагольного parser'а</title>
<style>
  :root { font-family: system-ui, -apple-system, sans-serif; color-scheme: light; }
  body { margin: 0; background: #f6f7f9; color: #20242a; }
  main { max-width: 1050px; margin: 0 auto; padding: 28px 20px 60px; }
  h1 { margin: 0 0 8px; font-size: 26px; }
  .sub { margin: 0 0 20px; color: #65707d; }
  textarea {
    box-sizing: border-box; width: 100%; min-height: 125px; resize: vertical;
    padding: 12px; border: 1px solid #b8c0ca; border-radius: 8px;
    background: white; font: 18px/1.45 ui-monospace, SFMono-Regular, monospace;
  }
  .bar { display: flex; gap: 10px; align-items: center; margin: 10px 0 22px; }
  button {
    border: 1px solid #89929d; border-radius: 7px; background: white;
    padding: 8px 14px; font: inherit; cursor: pointer;
  }
  button.primary { background: #20242a; color: white; border-color: #20242a; }
  .muted { color: #65707d; font-size: 14px; }
  .form {
    background: white; border: 1px solid #d8dde3; border-radius: 10px;
    padding: 16px; margin: 14px 0;
  }
  .form-head { display: flex; gap: 12px; align-items: baseline; flex-wrap: wrap; }
  .surface { font: 700 21px ui-monospace, SFMono-Regular, monospace; }
  .reason { margin: 5px 0 12px; }
  .suggestion {
    border-left: 4px solid #72a678; background: #f5faf5;
    padding: 9px 11px; margin: 8px 0;
  }
  .seg, .gloss { font: 17px/1.35 ui-monospace, SFMono-Regular, monospace; }
  .gloss { color: #39414a; }
  .meta { color: #65707d; font-size: 13px; margin-top: 4px; }
  details { margin-top: 10px; }
  summary { cursor: pointer; color: #4d5966; }
  table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 14px; }
  td, th { padding: 6px 7px; border-bottom: 1px solid #e3e6ea; text-align: left; }
  code { font-family: ui-monospace, SFMono-Regular, monospace; }
  .empty { background: #fff7ef; border-left: 4px solid #c58b50; padding: 9px 11px; }
</style>
</head>
<body>
<main>
  <h1>Тест глагольного parser'а</h1>
  <p class="sub">
    Это прямой вызов rule engine: здесь нет проверки token eligibility и ничего
    не записывается в тексты. Поэтому можно понять, проблема в правилах или в интерфейсе.
  </p>

  <textarea id="forms" autofocus
    placeholder="Вставь формы — по одной на строке или через пробел&#10;&#10;mzabhile&#10;kəmʕawaḷan"></textarea>

  <div class="bar">
    <button class="primary" id="run">Проверить</button>
    <button id="clear">Очистить</button>
    <span class="muted">Enter в textarea не запускает тест: можно вставлять список строк.</span>
  </div>

  <div id="status" class="muted"></div>
  <section id="results"></section>
</main>

<script>
const forms = document.getElementById("forms");
const results = document.getElementById("results");
const status = document.getElementById("status");

const esc = (s) => String(s ?? "")
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;");

function evidenceText(row) {
  const e = row.evidence;
  if (!e) return "без подтверждения во FLEx";
  return `${e.match}: ${e.stem || "—"} / ${e.gloss || "—"}${e.count ? ` ×${e.count}` : ""}`;
}

function renderOne(r) {
  const box = document.createElement("article");
  box.className = "form";

  let body = `
    <div class="form-head">
      <span class="surface">${esc(r.surface)}</span>
      <span class="muted">C: ${esc(r.skeleton || "∅")}</span>
    </div>
    <div class="reason">${esc(r.reason)}</div>
  `;

  if (r.suggestions.length) {
    body += r.suggestions.map((row) => `
      <div class="suggestion">
        <div class="seg">${esc(row.segmentation)}</div>
        <div class="gloss">${esc(row.gloss)}</div>
        <div class="meta">
          ${esc(row.confidence)} ·
          ${esc(row.rule_label || row.rule)} ·
          ${esc(evidenceText(row))}
        </div>
      </div>
    `).join("");
  } else {
    body += `<div class="empty">Parser не выдал ни одного анализа.</div>`;
  }

  if (r.suffix_matches.length) {
    body += `
      <details>
        <summary>Совпавшие правила окончаний (${r.suffix_matches.length})</summary>
        <table>
          <thead><tr>
            <th>суффикс</th><th>правило</th><th>stem</th><th>C</th><th>FLEx evidence</th>
          </tr></thead>
          <tbody>
          ${r.suffix_matches.map((x) => `
            <tr>
              <td><code>${esc(x.suffix)}</code></td>
              <td>${esc(x.label)}</td>
              <td><code>${esc(x.stem)}</code></td>
              <td><code>${esc(x.skeleton || "∅")}</code></td>
              <td>${x.evidence.length
                ? x.evidence.map(e => esc(`${e.stem}/${e.gloss} (${e.match})`)).join("<br>")
                : "—"}</td>
            </tr>
          `).join("")}
          </tbody>
        </table>
      </details>
    `;
  }

  if (r.preverb_matches.length) {
    body += `
      <details>
        <summary>Совпавшие превербы (${r.preverb_matches.length})</summary>
        ${r.preverb_matches.map((x) =>
          `<div><code>${esc(x.prefix)}=</code> ${esc(x.gloss)} → host <code>${esc(x.host)}</code></div>`
        ).join("")}
      </details>
    `;
  }

  box.innerHTML = body;
  return box;
}

async function run() {
  const list = forms.value
    .split(/\s+/u)
    .map(s => s.trim())
    .filter(Boolean);

  results.innerHTML = "";
  if (!list.length) {
    status.textContent = "Вставь хотя бы одну форму.";
    return;
  }

  status.textContent = `Проверяю ${list.length} форм…`;

  const response = await fetch("/api/test", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({forms: list})
  });

  if (!response.ok) {
    status.textContent = `Ошибка HTTP ${response.status}`;
    return;
  }

  const data = await response.json();
  status.textContent =
    `Готово: ${data.results.length} форм. FLEx evidence загружено: ${data.evidence_rows}.`;

  for (const row of data.results) results.append(renderOne(row));
}

document.getElementById("run").addEventListener("click", run);
document.getElementById("clear").addEventListener("click", () => {
  forms.value = "";
  results.innerHTML = "";
  status.textContent = "";
  forms.focus();
});

forms.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run();
});
</script>
</body>
</html>
"""


def create_web_app(parser: TuroyoVerbParser) -> FastAPI:
    app = FastAPI(title="Turoyo verb parser tester")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE

    @app.post("/api/test")
    def test_forms(body: dict[str, Any]) -> dict[str, Any]:
        raw = body.get("forms") or []
        if isinstance(raw, str):
            raw = raw.split()
        forms = [str(x).strip() for x in raw if str(x).strip()][:200]
        return {
            "evidence_rows": len(parser.evidence.rows),
            "results": [diagnose(parser, form) for form in forms],
        }

    return app


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Direct tester for the Turoyo verb rule engine."
    )
    ap.add_argument("forms", nargs="*", help="Forms to test in terminal mode")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    root = find_root()
    print(f"Project: {root}")
    print("Загружаю FLEx evidence…", flush=True)
    parser = make_parser(root)
    print(f"FLEx verbal evidence rows: {len(parser.evidence.rows)}", flush=True)

    if args.forms:
        for form in args.forms:
            print_result(diagnose(parser, form))
        return

    url = f"http://127.0.0.1:{args.port}"
    print(f"\nОткрой: {url}")
    print("Остановка: Ctrl+C\n")

    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    uvicorn.run(
        create_web_app(parser),
        host="127.0.0.1",
        port=args.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
