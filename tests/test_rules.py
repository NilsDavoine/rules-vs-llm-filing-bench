import pytest

from filing_bench import corpus, dates, guard
from filing_bench.rules import V0, V1, V3, VERSIONS, short_name
from filing_bench.schema import NO_DATE


def piece(text, filename="scan.pdf", company="Exemple Conseil"):
    return {"text": text, "filename": filename, "company": company}


@pytest.mark.parametrize("text,period", [
    ("Date de facture : 05.10.2025\nÉchéance : 20 octobre 2025", "2025-10"),
    ("Échéance : 20/11/2025\nDate de facture : 05/10/2025", "2025-10"),     # due date first in layout
    ("Période du 01/03/2025 au 31/03/2025\nImprimé le 02/04/2025", "2025-03"),
    ("Kontoauszug 4/2025\nvom 01.03.2025 bis 31.03.2025", "2025-03"),         # number looks like a month
    ("Ticket 1234\n14.03.25 12:41\nTOTAL EUR 12,00", "2025-03"),                # two-digit year
    ("TOTAL EUR 12,00", NO_DATE),                                               # nothing to read: say so
])
def test_v1_reads_the_right_date(text, period):
    assert V1().period(piece(text), "") == period


def test_v0_invents_a_date_and_the_guard_blocks_it():
    p = V0().propose(piece("TOTAL EUR 12,00", filename="IMG_20250601_101010.jpg"))
    assert p.period == "2025-06"                       # the photo's date, not the receipt's
    assert not guard.supported(p.period, "TOTAL EUR 12,00")


@pytest.mark.parametrize("split", ["dev", "test"])
def test_rules_from_v1_on_never_invent(split):
    for v in VERSIONS[1:]:
        for p in corpus.load(split):
            assert guard.supported(v.propose(p).period, p["text"]), (v.name, p["id"])


@pytest.mark.parametrize("raw,short", [
    ("PAPETERIE WEBER S.À R.L.", "Papeterie Weber"),
    ("Bürotechnik Kremer GmbH", "Bürotechnik Kremer"),
    ("GARAGE DU VIADUC S.À R.L.", "Garage du Viaduc"),
    ("Administration fiscale / Steuerverwaltung", "Administration fiscale"),
    ("Cloudvale Hosting Ltd", "Cloudvale Hosting"),
])
def test_short_name(raw, short):
    assert short_name(raw) == short


def test_v3_sees_a_sales_invoice_and_its_customer():
    text = "EXEMPLE CONSEIL S.A.\nFACTURE 2025-001\nFacturé à : HÔTEL BELVÉDÈRE S.A.\nDate : 03/02/2025"
    p = V3().propose(piece(text))
    assert (p.nature, p.emitter, p.folder) == ("Facture client", "Hôtel Belvédère", "2025/Ventes")


def test_v3_negative_total_is_a_credit_note():
    text = "TRAITEUR X S.A.\nFacture d'avoir 12\nDate : 03/02/2025\nTotal TTC : -120,00 EUR"
    assert V3().propose(piece(text)).nature == "Note de crédit"


def test_v3_finds_the_emitter_in_the_footer_when_the_logo_was_an_image():
    text = "www.papeterie-weber.lu\nFACTURE 25-1\nDate : 03/02/2025\n\nPapeterie Weber S.à r.l. - RCS B12345 - TVA LU1"
    assert V3().propose(piece(text)).emitter == "Papeterie Weber"


def test_dates_keep_positions_on_accented_text():
    text = "Échéance : 20 décembre 2025"
    m = dates.find(text)[0]
    assert text[m.start:m.end] == "20 décembre 2025"
