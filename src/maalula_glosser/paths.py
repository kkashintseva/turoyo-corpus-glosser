from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @classmethod
    def discover(cls, root: str | Path | None = None) -> "ProjectPaths":
        if root is not None:
            return cls(Path(root).resolve())
        current = Path.cwd().resolve()
        for candidate in (current, *current.parents):
            # ``ensure()`` is responsible for creating data/.  Requiring it
            # here makes a clean checkout impossible to start.
            if (candidate / "pyproject.toml").exists():
                return cls(candidate)
        raise FileNotFoundError("Could not discover project root; pass --root explicitly")

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def interim(self) -> Path:
        return self.root / "data" / "interim"

    @property
    def canonical(self) -> Path:
        return self.root / "data" / "canonical"

    @property
    def review(self) -> Path:
        return self.root / "data" / "review"

    @property
    def exports(self) -> Path:
        return self.root / "data" / "exports"

    @property
    def final(self) -> Path:
        return self.root / "data" / "final"

    @property
    def analysis_all(self) -> Path:
        return self.root / "data" / "analysis_all"

    @property
    def logs(self) -> Path:
        return self.root / "data" / "logs"

    def ensure(self) -> None:
        for path in (self.raw, self.interim, self.canonical, self.review, self.exports, self.final, self.analysis_all, self.logs):
            path.mkdir(parents=True, exist_ok=True)
