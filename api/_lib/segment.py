"""Sentence segmentation.

Deliberately a port of `lib/chunk.ts`, not an independent implementation.
The browser splits the document to build the heatmap and the Python function
splits it to score sentences; if the two disagree by even one boundary the
scores land on the wrong spans. Any change here needs the same change there.
"""

from __future__ import annotations

import re

ABBREVIATIONS: frozenset[str] = frozenset(
    """mr mrs ms dr prof sr jr st mt rev hon gen col lt sgt capt cmdr adm gov sen
    rep vs etc eg ie cf al approx dept est fig inc ltd co corp univ assn bros no
    vol pp ed eds trans misc min max avg std ca circa jan feb mar apr jun jul aug
    sep sept oct nov dec mon tue wed thu fri sat sun u.s u.k e.g i.e a.m p.m ph.d
    m.d b.a m.a""".split()
)

TERMINATORS = frozenset(".!?…")
CLOSERS = frozenset("\"'”’)]}»")

_TRAILING_WORD = re.compile(r"[A-Za-z.]+$")
_INITIAL = re.compile(r"(^|\s)[A-Za-z]$")
_OPENER = re.compile(r"[A-Z0-9\"'“‘(\[]")
_BLANK_LINE = re.compile(r"\n[ \t]*\n")


def _is_sentence_break(text: str, i: int) -> bool:
    ch = text[i]
    if ch not in TERMINATORS:
        return False

    j = i + 1
    while j < len(text) and text[j] in CLOSERS:
        j += 1

    if j < len(text) and not text[j].isspace():
        return False

    if ch == ".":
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if i > 0 and text[i - 1].isdigit() and nxt.isdigit():
            return False

        before = text[max(0, i - 12) : i]
        match = _TRAILING_WORD.search(before)
        word = (match.group(0) if match else "").lower().rstrip(".")
        if word in ABBREVIATIONS:
            return False

        if _INITIAL.search(before):
            return False

    k = j
    while k < len(text) and text[k].isspace():
        k += 1
    if k >= len(text):
        return True
    return bool(_OPENER.match(text[k])) or text[k] == "—"


def split_sentences_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Return ``(sentence, start, end)`` triples with offsets into `text`."""
    out: list[tuple[str, int, int]] = []
    cursor = 0

    def push(start: int, end: int) -> None:
        raw = text[start:end]
        lead = len(raw) - len(raw.lstrip())
        trail = len(raw) - len(raw.rstrip())
        s, e = start + lead, end - trail
        if e > s:
            out.append((text[s:e], s, e))

    i = 0
    while i < len(text):
        # A blank line is a hard break even without a terminator: headings and
        # list items must not be glued onto the sentence that follows.
        if text[i] == "\n" and _BLANK_LINE.match(text, i):
            push(cursor, i)
            j = i
            while j < len(text) and text[j].isspace():
                j += 1
            cursor = j
            i = j
            continue

        if _is_sentence_break(text, i):
            j = i + 1
            while j < len(text) and text[j] in CLOSERS:
                j += 1
            push(cursor, j)
            cursor = j
            i = j
            continue

        i += 1

    push(cursor, len(text))
    return out


def split_sentences(text: str) -> list[str]:
    return [s for s, _, _ in split_sentences_with_offsets(text)]
