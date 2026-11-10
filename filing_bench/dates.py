"""Reading dates in a document, with their position and the words before them.

Used by the rules engine from v1 on. (The guard has its own, simpler reader
on purpose.)
"""
from dataclasses import dataclass
import re
import unicodedata

_MONTHS = {
    "janvier": 1, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6, "juillet": 7,
    "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11, "decembre": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "januar": 1, "februar": 2, "marz": 3, "juni": 6, "juli": 7, "oktober": 10, "dezember": 12,
}
_M = "|".join(sorted(_MONTHS, key=len, reverse=True))


def plain(text):
    """Lower case, accents removed, SAME LENGTH as the input (positions stay valid)."""
    out = []
    for c in text or "":
        d = unicodedata.normalize("NFD", c)
        out.append(d[0].lower() if d else c)
    return "".join(out)


@dataclass
class Mention:
    start: int
    end: int
    year: int
    month: int          # 0 when only the year is known
    kind: str           # full | month | quarter | year
    line: int
    before: str         # the words just before, same line, plain text

    @property
    def period(self):
        return f"{self.year}" if not self.month else f"{self.year}-{self.month:02d}"


def _y(s):
    y = int(s)
    return 2000 + y if y < 100 else y


PATTERNS = [
    ("full", re.compile(r"(?<![\d./-])(\d{4})-(\d{2})-(\d{2})(?![\d])"), lambda m: (int(m[1]), int(m[2]))),
    ("full", re.compile(r"(?<![\d./-])(\d{1,2})[./-](\d{1,2})[./-](\d{4}|\d{2})(?![\d./-]?\d)"),
     lambda m: (_y(m[3]), int(m[2]))),
    ("full", re.compile(r"\b(%s)\s+(\d{1,2}),\s*(\d{4})\b" % _M), lambda m: (int(m[3]), _MONTHS[m[1]])),
    ("full", re.compile(r"(?<!\d)(\d{1,2})(?:er|\.)?\s+(%s)\s+(\d{4})\b" % _M), lambda m: (int(m[3]), _MONTHS[m[2]])),
    ("quarter", re.compile(r"\b(?:q|t)([1-4])\s*[/ ]\s*(\d{4})\b"), lambda m: (int(m[2]), 3 * int(m[1]))),
    ("quarter", re.compile(r"\b([1-4])\s*(?:er|e|eme|st|nd|rd|th|\.)?\s*(?:trimestre|quartal|quarter)\s+(\d{4})\b"),
     lambda m: (int(m[2]), 3 * int(m[1]))),
    ("quarter", re.compile(r"\bquarter\s+([1-4])\s+(\d{4})\b"), lambda m: (int(m[2]), 3 * int(m[1]))),
    ("month", re.compile(r"\b(%s)\s+(\d{4})\b" % _M), lambda m: (int(m[2]), _MONTHS[m[1]])),
    ("month", re.compile(r"(?<![\d./-])(\d{1,2})[./](\d{4})(?![\d./])"), lambda m: (int(m[2]), int(m[1]))),
    ("year", re.compile(r"(?<![\d./-])((?:19|20)\d{2})(?![\d./-]?\d)"), lambda m: (int(m[1]), 0)),
]


def find(text, window=40):
    """All dates of the text, longest reading first where two overlap."""
    t = plain(text)
    starts = [0]
    for i, c in enumerate(t):
        if c == "\n":
            starts.append(i + 1)
    taken = []
    out = []
    for kind, rx, read in PATTERNS:
        for m in rx.finditer(t):
            if any(a < m.end() and m.start() < b for a, b in taken):
                continue
            try:
                year, month = read(m)
            except (KeyError, ValueError):
                continue
            if not (1990 <= year <= 2035 and 0 <= month <= 12):
                continue
            if kind == "full" and month == 0:
                continue
            line = sum(1 for s in starts if s <= m.start()) - 1
            line_start = starts[line]
            before = t[max(line_start, m.start() - window):m.start()]
            out.append(Mention(m.start(), m.end(), year, month, kind, line, before))
            taken.append((m.start(), m.end()))
    out.sort(key=lambda x: x.start)
    return out
