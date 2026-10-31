"""Console helpers shared by the demo / analysis entry points."""

import sys


def utf8_stdout():
    """Make redirected stdout / stderr UTF-8.

    On Windows a redirected stream (a file, a pipe, a screen recorder) defaults to the ANSI code page (cp1252), which cannot
    encode the report's symbols (Δ, µ, ×) or arbitrary model output, so `print` raised UnicodeEncodeError mid-demo.
    Interactive consoles are left alone (Windows writes them as UTF-16 already)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure') and not stream.isatty():
            stream.reconfigure(encoding='utf-8', errors='replace')
