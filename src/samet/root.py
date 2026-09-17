"""Project-root discovery for experiment defaults."""

from __future__ import annotations

from pathlib import Path

from samet.exceptions import SametError


PROJECT_MARKERS = (".git", ".venv", "README.md")


class ProjectRootNotFoundError(SametError):
    """Raised when no project-root marker can be found."""


def resolve_project_root(root_dir: Path | None = None) -> Path:
    """Resolve an explicit root or discover one from the current directory.

    Marker priority is ``.git``, then ``.venv``, then ``README.md``. Within a
    marker class, the nearest ancestor wins.
    """
    if root_dir is not None:
        root = Path(root_dir).expanduser().resolve()
        if root.exists() and not root.is_dir():
            raise NotADirectoryError(root)
        return root

    start = Path.cwd().resolve()
    ancestors = (start, *start.parents)
    for marker in PROJECT_MARKERS:
        for candidate in ancestors:
            if (candidate / marker).exists():
                return candidate
    raise ProjectRootNotFoundError(
        "Could not discover the project root; pass root_dir explicitly."
    )
