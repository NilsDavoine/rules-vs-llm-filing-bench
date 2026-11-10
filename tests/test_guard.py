"""The guard: no date in a file name unless the document shows it."""
import pytest

from filing_bench import corpus, guard
from filing_bench.schema import NO_DATE, TO_SORT, Proposal


@pytest.mark.parametrize("text,period", [
    ("Date : 14/03/2025", "2025-03"),
    ("Datum: 14.03.2025", "2025-03"),
    ("Issued on March 14, 2025", "2025-03"),
    ("le 1er mars 2025", "2025-03"),
    ("14. März 2025", "2025-03"),
    ("ticket 14.03.25 12:41", "2025-03"),
    ("Période de cotisation : 02/2025", "2025-02"),
    ("2e trimestre 2024", "2024-06"),
    ("Q3/2024", "2024-08"),
    ("année d'imposition 2023", "2023"),
    ("no date at all", NO_DATE),
])
def test_supported(text, period):
    assert guard.supported(period, text)


@pytest.mark.parametrize("text,period", [
    ("Date : 14/03/2025", "2025-04"),
    ("Date : 14/03/2025", "2024-03"),
    ("Total 2 025,00 EUR", "2025-01"),
    ("no date at all", "2025"),
    ("anything", "March 2025"),
])
def test_not_supported(text, period):
    assert not guard.supported(period, text)


def test_file_name_is_not_evidence():
    piece = {"filename": "IMG_20250601_101010.jpg", "text": "TOTAL EUR 12,40\nMerci"}
    p, caught = guard.enforce(Proposal(period="2025-06", nature="Note de frais",
                                       folder="2025/Notes de frais"), piece["text"])
    assert caught and p.period == NO_DATE and p.folder == TO_SORT
    assert p.name == "SANS-DATE Note de frais"


@pytest.mark.parametrize("split", ["dev", "test"])
def test_a_hallucinating_model_never_gets_a_date_through(split):
    """A proposer that always answers a date no piece shows: after the guard,
    every single name must say SANS-DATE and no piece may sit in a year folder."""
    for piece in corpus.load(split):
        lie = Proposal(period="1999-07", nature="Facture", emitter="X", folder="1999/Achats")
        p, caught = guard.enforce(lie, piece["text"])
        assert caught, piece["id"]
        assert p.name.startswith("SANS-DATE "), piece["id"]
        assert p.folder == TO_SORT, piece["id"]


def test_honest_answers_pass_untouched():
    """Every expected answer of the corpus is backed by its text: the guard
    never punishes the truth."""
    for split in ("dev", "test"):
        for piece in corpus.load(split):
            e = piece["expected"]
            p, caught = guard.enforce(Proposal(period=e["period"], folder=e["folder"]), piece["text"])
            assert not caught, (piece["id"], e["period"])
            assert p.folder == e["folder"]
