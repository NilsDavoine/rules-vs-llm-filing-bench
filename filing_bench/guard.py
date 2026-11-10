"""The guard: never put a date in a file name that the document does not show.

A wrong nature is visible and gets corrected. An invented date is worse: the
file lands in the wrong year folder, looks plausible, and nobody looks again.
So whatever proposes a period (rules, a local model, Claude), the period goes
through `enforce` before it becomes a name:

  * "YYYY-MM" passes only if the text mentions that month of that year
    (any common format, FR/EN/DE, numeric or in words, or a quarter that
    contains it);
  * "YYYY" passes only if the text mentions that year;
  * otherwise the period becomes SANS-DATE ("no date") and a year folder,
    which would be just as invented, becomes the "to sort" folder.

The file NAME is not evidence: "IMG_20250601.jpg" is the day the photo was
taken, not the day of the invoice.

This module is deliberately independent of the rules engine, so that a bug
in the engine's date reader cannot hide itself here.
"""
import re
import unicodedata

from .schema import NO_DATE, PERIOD_RE, TO_SORT

_MONTHS = {
    # French, English, German, accents removed, with common short forms
    "janvier": 1, "janv": 1, "january": 1, "jan": 1, "januar": 1, "janner": 1,
    "fevrier": 2, "fevr": 2, "february": 2, "feb": 2, "februar": 2,
    "mars": 3, "march": 3, "mar": 3, "marz": 3,
    "avril": 4, "april": 4, "apr": 4, "avr": 4,
    "mai": 5, "may": 5,
    "juin": 6, "june": 6, "jun": 6, "juni": 6,
    "juillet": 7, "july": 7, "jul": 7, "juli": 7, "juil": 7,
    "aout": 8, "august": 8, "aug": 8,
    "septembre": 9, "september": 9, "sep": 9, "sept": 9,
    "octobre": 10, "october": 10, "oct": 10, "oktober": 10, "okt": 10,
    "novembre": 11, "november": 11, "nov": 11,
    "decembre": 12, "december": 12, "dec": 12, "dezember": 12, "dez": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))


def _plain(text):
    text = unicodedata.normalize("NFD", text or "")
    return "".join(c for c in text if unicodedata.category(c) != "Mn").lower()


def _year(y):
    y = int(y)
    return 2000 + y if y < 100 else y


def mentioned(text):
    """Every (year, month) the text mentions; month is None for a bare year."""
    t = _plain(text)
    found = set()
    for m in re.finditer(r"(?<!\d)(\d{1,2})[./-](\d{1,2})[./-](\d{4}|\d{2})(?!\d)", t):
        d, mo, y = int(m.group(1)), int(m.group(2)), _year(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            found.add((y, mo))
    for m in re.finditer(r"(?<!\d)(\d{4})-(\d{2})(?:-(\d{2}))?(?!\d)", t):
        if 1 <= int(m.group(2)) <= 12:
            found.add((int(m.group(1)), int(m.group(2))))
    for m in re.finditer(r"(?<![\d./])(\d{1,2})[./](\d{4})(?!\d)", t):
        if 1 <= int(m.group(1)) <= 12:
            found.add((int(m.group(2)), int(m.group(1))))
    for m in re.finditer(r"\b(%s)\b\.?\s*(?:\d{1,2}(?:st|nd|rd|th)?,?\s*)?(\d{4})\b" % _MONTH_ALT, t):
        found.add((int(m.group(2)), _MONTHS[m.group(1)]))
    for m in re.finditer(r"\b(?:q|t)([1-4])\s*[/ ]?\s*(\d{4})\b"
                         r"|\b([1-4])\s*(?:er|e|eme|st|nd|rd|th|\.)?\s*(?:trimestre|quartal|quarter)\s*(\d{4})\b"
                         r"|\bquarter\s*([1-4])\s*(\d{4})\b", t):
        q = int(m.group(1) or m.group(3) or m.group(5))
        y = int(m.group(2) or m.group(4) or m.group(6))
        for mo in range(3 * q - 2, 3 * q + 1):
            found.add((y, mo))
    for m in re.finditer(r"(?<!\d)((?:19|20)\d{2})(?!\d)", t):
        found.add((int(m.group(1)), None))
    return found


def supported(period, text):
    """True when the text backs the period. SANS-DATE is always honest."""
    if period == NO_DATE:
        return True
    m = PERIOD_RE.match(period or "")
    if not m:
        return False
    marks = mentioned(text)
    year = int(m.group(1))
    if m.group(2) is None:
        return any(y == year for y, _ in marks)
    return (year, int(m.group(2))) in marks


def enforce(proposal, text):
    """Return (proposal, caught). `caught` is True when a date was invented."""
    if supported(proposal.period, text):
        return proposal, False
    proposal.period = NO_DATE
    if re.match(r"^\d{4}/", proposal.folder or ""):
        proposal.folder = TO_SORT
    return proposal, True
