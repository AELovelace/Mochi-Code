# -*- coding: utf-8 -*-
"""Detached Sakura-themed text editor window."""
from __future__ import annotations

import os

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QCloseEvent, QFont, QKeySequence
from PyQt6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QStatusBar,
    QToolBar,
)

from .editor_support import (
    detect_line_ending,
    normalize_for_editor,
    read_utf8_text,
    serialize_for_disk,
    write_utf8_text,
)


class EditorWindow(QMainWindow):
    """A small detached editor for opening and saving UTF-8 text files."""

    def __init__(self, file_path: str | None = None) -> None:
        super().__init__(None)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        self._current_path: str | None = None
        self._dirty = False
        self._line_ending = "\n"
        self._suspend_text_events = False

        self.resize(860, 640)
        self._build_ui()
        self._apply_theme()
        self._replace_editor_text("")
        self._refresh_window_state()

        if file_path:
            self.load_path(file_path)

    @property
    def current_path(self) -> str | None:
        return self._current_path

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def _build_ui(self) -> None:
        toolbar = QToolBar("Editor Actions", self)
        toolbar.setMovable(False)
        toolbar.setObjectName("editorToolbar")
        self.addToolBar(toolbar)

        open_action = QAction("Open", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_from_dialog)
        toolbar.addAction(open_action)

        save_action = QAction("Save", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_file)
        toolbar.addAction(save_action)

        save_as_action = QAction("Save As", self)
        save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_as_action.triggered.connect(self.save_file_as)
        toolbar.addAction(save_as_action)

        self._editor = QPlainTextEdit()
        self._editor.setObjectName("editorSurface")
        self._editor.setTabStopDistance(32)
        self._editor.setPlaceholderText("Open a file, or start writing here...")
        self._editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._editor.setFont(QFont("Consolas", 11))
        self._editor.textChanged.connect(self._on_text_changed)
        self.setCentralWidget(self._editor)

        status_bar = QStatusBar(self)
        status_bar.setObjectName("editorStatusBar")
        self.setStatusBar(status_bar)

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #f8e9ee;
                color: #5e3f49;
            }
            QToolBar#editorToolbar {
                spacing: 8px;
                padding: 8px 12px;
                background: rgba(255, 248, 250, 0.96);
                border: none;
                border-bottom: 1px solid rgba(193, 142, 156, 0.35);
            }
            QToolBar#editorToolbar QToolButton {
                color: #764d59;
                background: rgba(255, 233, 239, 0.94);
                border: 1px solid rgba(193, 142, 156, 0.55);
                border-radius: 14px;
                padding: 6px 12px;
                font-weight: 600;
            }
            QToolBar#editorToolbar QToolButton:hover {
                background: rgba(255, 245, 248, 0.98);
            }
            QPlainTextEdit#editorSurface {
                background: rgba(255, 251, 252, 0.95);
                color: #38242c;
                border: 1px solid rgba(193, 142, 156, 0.28);
                border-radius: 18px;
                padding: 14px;
                selection-background-color: rgba(191, 118, 141, 0.35);
            }
            QStatusBar#editorStatusBar {
                background: rgba(255, 248, 250, 0.96);
                color: #734a56;
                border-top: 1px solid rgba(193, 142, 156, 0.28);
            }
            QScrollBar:vertical, QScrollBar:horizontal {
                background: rgba(255, 246, 248, 0.55);
                margin: 2px;
                border-radius: 5px;
            }
            QScrollBar:vertical {
                width: 10px;
            }
            QScrollBar:horizontal {
                height: 10px;
            }
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
                background: rgba(191, 118, 141, 0.65);
                min-height: 24px;
                min-width: 24px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: transparent;
                width: 0px;
                height: 0px;
            }
            """
        )

    def _on_text_changed(self) -> None:
        if self._suspend_text_events:
            return
        if not self._dirty:
            self._set_dirty(True)

    def _set_dirty(self, dirty: bool) -> None:
        self._dirty = dirty
        self._refresh_window_state()

    def _replace_editor_text(self, content: str) -> None:
        self._suspend_text_events = True
        try:
            self._editor.setPlainText(content)
        finally:
            self._suspend_text_events = False
        self._editor.document().setModified(False)

    def _refresh_window_state(self) -> None:
        name = os.path.basename(self._current_path) if self._current_path else "Untitled"
        marker = " *" if self._dirty else ""
        self.setWindowTitle(f"{name}{marker} - Sakura Editor")

        if self._current_path:
            state = "Modified" if self._dirty else "Ready"
            self.statusBar().showMessage(f"{state}  |  {self._current_path}")
        elif self._dirty:
            self.statusBar().showMessage("Modified  |  Unsaved file")
        else:
            self.statusBar().showMessage("Ready  |  Unsaved file")

    def _show_file_error(self, action: str, path: str, exc: Exception) -> None:
        QMessageBox.critical(
            self,
            f"{action} failed",
            f"Could not {action.lower()}:\n{path}\n\n{exc}",
        )

    def _confirm_replace_if_dirty(self) -> bool:
        if not self._dirty:
            return True

        reply = QMessageBox.question(
            self,
            "Unsaved changes",
            "Save your changes before opening another file?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if reply == QMessageBox.StandardButton.Save:
            return self.save_file()
        return reply == QMessageBox.StandardButton.Discard

    def open_from_dialog(self) -> bool:
        if not self._confirm_replace_if_dirty():
            return False

        start_dir = self._current_path or os.getcwd()
        path, _ = QFileDialog.getOpenFileName(self, "Open File", start_dir, "All Files (*)")
        if not path:
            return False
        return self.load_path(path)

    def load_path(self, path: str) -> bool:
        try:
            content = read_utf8_text(path)
        except Exception as exc:
            self._show_file_error("Open", path, exc)
            return False

        self._current_path = os.path.abspath(path)
        self._line_ending = detect_line_ending(content)
        self._replace_editor_text(normalize_for_editor(content))
        self._set_dirty(False)
        self.statusBar().showMessage(f"Opened  |  {self._current_path}", 3000)
        return True

    def save_file(self) -> bool:
        if not self._current_path:
            return self.save_file_as()

        content = serialize_for_disk(self._editor.toPlainText(), self._line_ending)
        try:
            write_utf8_text(self._current_path, content)
        except Exception as exc:
            self._show_file_error("Save", self._current_path, exc)
            return False

        self._set_dirty(False)
        self.statusBar().showMessage(f"Saved  |  {self._current_path}", 3000)
        return True

    def save_file_as(self) -> bool:
        start_path = self._current_path or os.getcwd()
        path, _ = QFileDialog.getSaveFileName(self, "Save File As", start_path, "All Files (*)")
        if not path:
            return False

        self._current_path = os.path.abspath(path)
        if self.save_file():
            return True
        return False

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if not self._dirty:
            super().closeEvent(event)
            return

        reply = QMessageBox.question(
            self,
            "Unsaved changes",
            "Save your changes before closing this editor?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if reply == QMessageBox.StandardButton.Save:
            if self.save_file():
                super().closeEvent(event)
            else:
                event.ignore()
            return
        if reply == QMessageBox.StandardButton.Discard:
            super().closeEvent(event)
            return
        event.ignore()
