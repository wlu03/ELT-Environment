from __future__ import annotations

import unittest

from elt_environment.artifact import ArtifactFormatError, format_artifact, parse_artifact


class ParseArtifactTests(unittest.TestCase):
    def test_round_trip_keeps_content_byte_for_byte(self) -> None:
        files = {
            "main.tf": 'terraform {\n  required_providers {}\n}\n',
            "models/sources.yml": "version: 2\n",
            "models/a.sql": "select 1 as x",
        }
        self.assertEqual(parse_artifact(format_artifact(files)), files)

    def test_text_outside_blocks_is_ignored(self) -> None:
        reply = 'Here is the project.\n<file path="elt/main.tf">\nx = 1\n</file>\nDone.'
        self.assertEqual(parse_artifact(reply), {"main.tf": "x = 1"})

    def test_refusals(self) -> None:
        cases = {
            "no_file_blocks": "no blocks here",
            "malformed_file_block": '<file path="elt/a.sql">\nselect 1\n',
            "path_outside_elt": '<file path="main.tf">\nx\n</file>',
            "duplicate_path": '<file path="elt/a.sql">\n1\n</file>\n<file path="elt/a.sql">\n2\n</file>',
        }
        for reason, reply in cases.items():
            with self.subTest(reason=reason), self.assertRaises(ArtifactFormatError) as raised:
                parse_artifact(reply)
            self.assertEqual(str(raised.exception), reason)

    def test_paths_that_could_leave_the_project_root_are_refused(self) -> None:
        for path in (
            "elt/../outside",
            "elt/models/../../outside",
            "elt/./main.tf",
            "elt//etc/passwd",
            "elt/models/",
            "elt/models\\a.sql",
            "elt/models/a\x00.sql",
        ):
            with self.subTest(path=path), self.assertRaises(ArtifactFormatError) as raised:
                parse_artifact(f'<file path="{path}">\nx\n</file>')
            self.assertEqual(str(raised.exception), "unsafe_path")

    def test_nested_model_paths_are_accepted(self) -> None:
        files = parse_artifact('<file path="elt/models/marts/a.sql">\nselect 1\n</file>')
        self.assertEqual(files, {"models/marts/a.sql": "select 1"})

    def test_unterminated_second_block_refuses_the_whole_reply(self) -> None:
        reply = '<file path="elt/a.sql">\n1\n</file>\n<file path="elt/b.sql">\n2\n'
        with self.assertRaises(ArtifactFormatError):
            parse_artifact(reply)


if __name__ == "__main__":
    unittest.main()
