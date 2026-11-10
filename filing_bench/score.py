"""Scoring one proposal against the expected answer.

  period   exact period (YYYY-MM, YYYY or SANS-DATE)
  nature   the nature alone, without the emitter
  name     the whole file name (case, accents and punctuation ignored)
  folder   the exact folder of the tree
  ok       name AND folder right: the line can be accepted without editing
  invented the raw proposal carried a date the text does not show (caught by the guard)
"""
from . import guard
from .schema import compose_name, fold


def score(piece, proposal, invented):
    e = piece["expected"]
    expected_name = compose_name(e["period"], e["nature"], e["emitter"])
    r = {
        "period": proposal.period == e["period"],
        "nature": proposal.nature == e["nature"],
        "name": fold(proposal.name) == fold(expected_name),
        "folder": proposal.folder == e["folder"],
        "invented": invented,
    }
    r["ok"] = r["name"] and r["folder"]
    r["expected_name"] = expected_name
    return r


def guarded(piece, proposal):
    """Run the guard, then score. Returns (proposal, result)."""
    proposal, invented = guard.enforce(proposal, piece["text"])
    return proposal, score(piece, proposal, invented)


METRICS = ["period", "nature", "name", "folder", "ok", "invented"]


def summary(results):
    n = len(results) or 1
    return {k: sum(1 for r in results if r[k]) / n for k in METRICS}
