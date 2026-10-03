"""Console setup for a multilingual project on Windows.

The Windows console defaults to a legacy code page (cp1252 on this machine), which
raises ``UnicodeEncodeError`` on Devanagari, Tamil, and even on box-drawing
characters.  For a project whose subject matter *is* Hindi and Tamil text, a
crash on printing is not acceptable, so every entry point calls ``setup_console()``
first.

Strategy: reconfigure the streams to UTF-8 where possible; where it is not, fall
back to replacement characters rather than raising.  A garbled glyph in a
progress line is a cosmetic problem; a crashed twelve-hour batch job is not.
"""

from __future__ import annotations

import io
import os
import sys

_configured = False


def setup_console() -> bool:
    """Make stdout/stderr safe for arbitrary Unicode. Returns True if UTF-8 is active.

    Idempotent -- safe to call from every script.
    """
    global _configured
    if _configured:
        return supports_unicode()

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            # Python 3.7+: retarget the text wrapper without replacing the stream.
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            try:
                buffer = getattr(stream, "buffer", None)
                if buffer is not None:
                    setattr(
                        sys,
                        stream_name,
                        io.TextIOWrapper(buffer, encoding="utf-8", errors="replace", line_buffering=True),
                    )
            except (AttributeError, ValueError, OSError):
                pass  # Nothing more to try; safe_text() still protects callers.

    if sys.platform == "win32":
        # Enable ANSI escape processing so bold/colour render instead of leaking
        # literal escape codes into the output.
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            for handle_id in (-11, -12):  # STDOUT, STDERR
                handle = kernel32.GetStdHandle(handle_id)
                mode = ctypes.c_ulong()
                if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                    kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            pass

    _configured = True
    return supports_unicode()


def supports_unicode() -> bool:
    enc = (getattr(sys.stdout, "encoding", "") or "").lower()
    return "utf" in enc


def safe_text(text: str) -> str:
    """Return ``text`` rendered safely for the current stdout encoding.

    Use for content that may contain Devanagari or Tamil when writing to a
    console whose encoding could not be upgraded.
    """
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(enc)
        return text
    except (UnicodeEncodeError, LookupError):
        return text.encode(enc, errors="replace").decode(enc, errors="replace")


def use_ascii_only() -> bool:
    """Whether to draw with ASCII instead of box-drawing characters."""
    if os.environ.get("PRAMANA_ASCII", "").strip().lower() in {"1", "true", "yes"}:
        return True
    return not supports_unicode()


def rule(width: int = 72) -> str:
    return ("-" if use_ascii_only() else "─") * width


def bold(text: str) -> str:
    """Bold ``text`` only when stdout is an interactive terminal.

    Piping to a file or another process must not embed escape codes -- otherwise
    every captured log and redirected report is littered with ``[1m``.
    """
    if use_ascii_only() or not sys.stdout.isatty():
        return text
    return f"\033[1m{text}\033[0m"
