"""Parse a model reply into the files of an ``elt/`` artifact, and format files as a reply."""

from __future__ import annotations

import re
from collections.abc import Mapping

ARTIFACT_PREFIX = "elt/"
_FILE_BLOCK = re.compile(r'<file path="([^"\n]+)">\n(.*?)\n</file>', re.DOTALL)
_OPEN_TAG = re.compile(r"<file\b")
_CLOSE_TAG = re.compile(r"</file>")


class ArtifactFormatError(ValueError):
    """The reply does not contain a well-formed set of file blocks."""


def parse_artifact(text: str) -> dict[str, str]:
    """Return ``{path relative to elt/: content}`` for every file block in ``text``.

    Text outside the blocks is ignored. Content inside a block is returned
    unchanged. A reply with no block, an unterminated block, a path outside
    ``elt/``, or a repeated path is refused.
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
        if relative in files:
            raise ArtifactFormatError("duplicate_path")
        files[relative] = content
    return files


def format_artifact(files: Mapping[str, str]) -> str:
    """Render ``{path relative to elt/: content}`` as the reply format ``parse_artifact`` reads."""

    return "\n".join(
        f'<file path="{ARTIFACT_PREFIX}{path}">\n{content}\n</file>' for path, content in files.items()
    )
