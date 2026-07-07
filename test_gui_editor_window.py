import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from gui.editor_window import EditorWindow


class EditorWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def tearDown(self):
        for window in self._app.topLevelWidgets():
            if isinstance(window, EditorWindow):
                window._set_dirty(False)
                window.close()
        self._app.processEvents()

    def test_new_window_starts_clean(self):
        window = EditorWindow()
        self.assertFalse(window.is_dirty)
        self.assertIsNone(window.current_path)
        self.assertIn("Untitled - Sakura Editor", window.windowTitle())

    def test_text_edit_marks_window_dirty(self):
        window = EditorWindow()
        window._editor.insertPlainText("petals")
        self._app.processEvents()
        self.assertTrue(window.is_dirty)
        self.assertIn("* - Sakura Editor", window.windowTitle())

    def test_load_path_starts_clean_and_save_clears_dirty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "flower.txt")
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write("hello\n")

            window = EditorWindow()
            self.assertTrue(window.load_path(path))
            self.assertFalse(window.is_dirty)

            window._editor.setPlainText(window._editor.toPlainText() + "world")
            self._app.processEvents()
            self.assertTrue(window.is_dirty)
            self.assertTrue(window.save_file())
            self.assertFalse(window.is_dirty)

            with open(path, "r", encoding="utf-8", newline="") as handle:
                self.assertEqual(handle.read(), "hello\nworld")


if __name__ == "__main__":
    unittest.main()
