"""Error types for the QRL surface language, with source-caret rendering."""

from __future__ import annotations


class QRLError(Exception):
    """Base class for all surface-language errors."""

    def __init__(self, message: str, line: int, col: int, src: str | None = None):
        self.message = message
        self.line = line
        self.col = col
        self.src = src
        super().__init__(self._render())

    def _render(self) -> str:
        head = f"{self.kind}: {self.message} (line {self.line}, col {self.col})"
        if not self.src:
            return head
        lines = self.src.splitlines()
        if not (1 <= self.line <= len(lines)):
            return head
        gutter = f"{self.line:>4} | "
        caret = " " * (len(gutter) + max(self.col - 1, 0)) + "^"
        return f"{head}\n{gutter}{lines[self.line - 1]}\n{caret}"

    def with_source(self, src: str) -> "QRLError":
        """Return a copy of this error carrying `src`, so the caret renders."""
        return type(self)(self.message, self.line, self.col, src)

    @property
    def kind(self) -> str:
        return "error"


class QRLSyntaxError(QRLError):
    @property
    def kind(self) -> str:
        return "syntax error"


class QRLTypeError(QRLError):
    @property
    def kind(self) -> str:
        return "type error"
