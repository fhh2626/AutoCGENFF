"""Setuptools commands used only when building a distribution."""

from pathlib import Path

from setuptools.command.sdist import sdist


class CleanSdist(sdist):
    """Omit generated egg-info, including setuptools' appended SOURCES.txt."""

    def make_release_tree(self, base_dir: str, files: list[str]) -> None:
        sources = [
            name for name in files
            if not any(part.endswith(".egg-info") for part in Path(name).parts)
        ]
        super().make_release_tree(base_dir, sources)
