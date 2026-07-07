import os
import tempfile
import unittest

from gui.editor_support import (
    detect_line_ending,
    normalize_for_editor,
    read_utf8_text,
    resolve_explorer_root,
    serialize_for_disk,
    write_utf8_text,
)


class EditorSupportTests(unittest.TestCase):
    def test_resolve_explorer_root_uses_existing_configured_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            resolved = resolve_explorer_root(temp_dir, fallback_cwd="C:\\fallback")
        self.assertEqual(resolved, os.path.abspath(temp_dir))

    def test_resolve_explorer_root_falls_back_when_configured_path_missing(self):
        with tempfile.TemporaryDirectory() as fallback_dir:
            resolved = resolve_explorer_root("C:\\does-not-exist-sakura", fallback_dir)
        self.assertEqual(resolved, os.path.abspath(fallback_dir))

    def test_read_and_write_utf8_text_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "petals.txt")
            write_utf8_text(path, "sakura\nblossom")
            self.assertEqual(read_utf8_text(path), "sakura\nblossom")

    def test_read_utf8_text_raises_on_invalid_utf8(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "binary.dat")
            with open(path, "wb") as handle:
                handle.write(b"\xff\xfe\x00\x00")
            with self.assertRaises(UnicodeDecodeError):
                read_utf8_text(path)

    def test_write_utf8_text_raises_when_parent_directory_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "missing", "petals.txt")
            with self.assertRaises(FileNotFoundError):
                write_utf8_text(path, "hello")

    def test_line_ending_helpers_preserve_style(self):
        disk_text = "one\r\ntwo\r\n"
        self.assertEqual(detect_line_ending(disk_text), "\r\n")
        normalized = normalize_for_editor(disk_text)
        self.assertEqual(normalized, "one\ntwo\n")
        self.assertEqual(serialize_for_disk(normalized, "\r\n"), disk_text)


if __name__ == "__main__":
    unittest.main()
