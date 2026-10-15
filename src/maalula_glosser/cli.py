from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
import uvicorn

app = typer.Typer(
    no_args_is_help=True,
    help="Turoyo corpus gloss editor: FLEx corpus + FieldWorks LIFT lexicon.",
)


def _root(root: Optional[Path]) -> Path:
    if root is not None:
        project = root.expanduser().resolve()
    else:
        current = Path.cwd().resolve()
        project = next(
            (candidate for candidate in (current, *current.parents)
             if (candidate / "pyproject.toml").is_file()),
            None,
        )
        if project is None:
            raise typer.BadParameter(
                "Could not find the project root (pyproject.toml); pass --root."
            )
    (project / "data").mkdir(parents=True, exist_ok=True)
    return project


@app.command("edit")
def edit(
    root: Optional[Path] = typer.Option(None, help="Project root"),
    host: str = typer.Option("127.0.0.1", help="Bind address; keep local unless needed"),
    port: int = typer.Option(8000),
) -> None:
    """Run the interactive Turoyo gloss editor."""
    from maalula_glosser.editor.app import create_app

    uvicorn.run(create_app(_root(root)), host=host, port=port)


@app.command("import-flextext")
def import_flextext_cli(
    source: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, readable=True,
        help="FLEx .flextext/.xml corpus export",
    ),
    root: Optional[Path] = typer.Option(None, help="Project root"),
) -> None:
    """Import FLEx texts as editable documents and corpus evidence."""
    from maalula_glosser.flex_import import import_flextext_to_state

    try:
        result = import_flextext_to_state(source, _root(root))
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result["report"], ensure_ascii=False, indent=2))


@app.command("flex-status")
def flex_status_cli(
    root: Optional[Path] = typer.Option(None, help="Project root"),
) -> None:
    """Show counts in the imported FLEx corpus index."""
    from maalula_glosser.flex_corpus import FlexCorpusStore

    typer.echo(json.dumps(
        FlexCorpusStore(_root(root)).manifest(), ensure_ascii=False, indent=2
    ))


@app.command("import-lift")
def import_lift_cli(
    source: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, readable=True,
        help="FieldWorks .lift lexicon export",
    ),
    ranges: Optional[Path] = typer.Option(
        None, "--ranges", exists=True, file_okay=True, dir_okay=False,
        readable=True, help="Optional companion .lift-ranges file",
    ),
    root: Optional[Path] = typer.Option(None, help="Project root"),
) -> None:
    """Import FieldWorks LIFT as the primary lexical-entry dictionary."""
    from maalula_glosser.editor.store import SessionStore
    from maalula_glosser.lift_lexicon import LiftLexiconStore, backfill_editor_sessions

    project = _root(root)
    try:
        report = LiftLexiconStore(project).import_file(source, ranges=ranges)
        linking = backfill_editor_sessions(project, session_store=SessionStore(project))
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(
        {"lexicon": report, "linking": linking}, ensure_ascii=False, indent=2
    ))


@app.command("lift-status")
def lift_status_cli(
    root: Optional[Path] = typer.Option(None, help="Project root"),
) -> None:
    """Show counts/categories in the imported FieldWorks LIFT lexicon."""
    from maalula_glosser.lift_lexicon import LiftLexiconStore

    typer.echo(json.dumps(
        LiftLexiconStore(_root(root)).manifest(), ensure_ascii=False, indent=2
    ))


@app.command("ritter-status")
def ritter_status_cli() -> None:
    """Show statistics for the bundled Ritter dictionary index."""
    from maalula_glosser.ritter_dictionary import default_ritter_dictionary

    typer.echo(json.dumps(
        default_ritter_dictionary().status(), ensure_ascii=False, indent=2
    ))


@app.command("ritter-lookup")
def ritter_lookup_cli(
    form: str = typer.Argument(..., help="Turoyo wordform to look up"),
    limit: int = typer.Option(12, min=1, max=100, help="Maximum candidates"),
    exact_only: bool = typer.Option(
        False, "--exact-only", help="Disable same-consonant-skeleton suggestions"
    ),
) -> None:
    """Look up a form in Ritter; consonants stay strict, vowels may vary."""
    from maalula_glosser.ritter_dictionary import (
        consonant_skeleton,
        default_ritter_dictionary,
    )

    dictionary = default_ritter_dictionary()
    matches = dictionary.lookup(
        form,
        limit=limit,
        include_vowel_variants=not exact_only,
    )
    typer.echo(json.dumps(
        {
            "query": form,
            "consonant_skeleton": consonant_skeleton(form),
            "results": [match.as_dict() for match in matches],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    app()
