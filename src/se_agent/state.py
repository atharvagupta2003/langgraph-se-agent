import operator

from dataclasses import dataclass, field
from typing import Annotated



@dataclass(kw_only=True)
class FileSummary:
    filepath: str
    """Github file path to be processed."""
    
    summary: str
    """Summary of the file."""


@dataclass(kw_only=True)
class OnboardState:
    filepaths: Annotated[list, operator.add] = field(default_factory=list)
    """List of github file paths to be processed."""

    file_summaries: Annotated[list[FileSummary], operator.add] = field(default_factory=list)
    """List of file summaries."""


@dataclass(kw_only=True)
class FilepathState:
    filepath: str
    """Github file path to be processed."""


@dataclass(kw_only=True)
class Package:
    package_id: int
    """Storage ID of the package."""

    name: str
    """Name of the package."""

    filepaths: list[str] = field(default_factory=list)
    """List of filepaths. Defaults to an empty list."""

    summary: str | None = field(default=None)
    """Summary of the package (optional)."""
