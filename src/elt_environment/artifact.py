"""Parse a model reply into the files of an ``elt/`` artifact, and format files as a reply."""

from __future__ import annotations

import re
from collections.abc import Mapping

ARTIFACT_PREFIX = "elt/"
_FILE_BLOCK = re.compile(r'^<file path="([^"\n]+)">\n(.*?)\n</file>$', re.DOTALL | re.MULTILINE)
_OPEN_TAG = re.compile(r"^<file\b", re.MULTILINE)
_CLOSE_TAG = re.compile(r"^</file>$", re.MULTILINE)
_UNSAFE_SEGMENTS = frozenset({"", ".", ".."})


class ArtifactFormatError(ValueError):
    """The reply does not contain a well-formed set of file blocks."""


def _safe_relative_path(relative: str) -> bool:
    if "\\" in relative or any(ord(character) < 32 or ord(character) == 127 for character in relative):
        return False
    return not any(segment in _UNSAFE_SEGMENTS for segment in relative.split("/"))


def parse_artifact(text: str) -> dict[str, str]:
    """Return ``{path relative to elt/: content}`` for every file block in ``text``.

    Text outside the blocks is ignored, and a tag counts only at the start of
    a line, so prose may mention ``<file>``. Content inside a block is returned
    unchanged. A reply with no block, an unterminated block, a path outside
    ``elt/``, a path with an empty, ``.`` or ``..`` segment, a backslash or a
    control character, or a repeated path is refused.
    """

    blocks = _FILE_BLOCK.findall(text)
    if len(_OPEN_TAG.findall(text)) != len(blocks) or len(_CLOSE_TAG.findall(text)) != len(
        blocks
    ):
        raise ArtifactFormatError("malformed_file_block")
    if not blocks:
        raise ArtifactFormatError("no_file_blocks")
    files: dict[str, str] = {}
    for path, content in blocks:
        if not path.startswith(ARTIFACT_PREFIX) or path == ARTIFACT_PREFIX:
            raise ArtifactFormatError("path_outside_elt")
        relative = path[len(ARTIFACT_PREFIX) :]
        if not _safe_relative_path(relative):
            raise ArtifactFormatError("unsafe_path")
        if relative in files:
            raise ArtifactFormatError("duplicate_path")
        files[relative] = content
    return files


def format_artifact(files: Mapping[str, str]) -> str:
    """Render ``{path relative to elt/: content}`` as the reply format ``parse_artifact`` reads."""

    return "\n".join(
        f'<file path="{ARTIFACT_PREFIX}{path}">\n{content}\n</file>' for path, content in files.items()
    )
