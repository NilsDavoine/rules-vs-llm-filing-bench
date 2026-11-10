from filing_bench.llm import messages, parse
from filing_bench.schema import NO_DATE, TO_SORT


def test_parse_clean_answer():
    p = parse('{"period": "2025-03", "nature": "Facture", "emitter": "Papeterie Weber", "folder": "Achats"}')
    assert (p.name, p.folder) == ("2025-03 Facture Papeterie Weber", "2025/Achats")


def test_parse_tolerates_text_around_and_accent_case():
    p = parse('Sure!\n```json\n{"period": "2024", "nature": "bulletin d\'imposition", "emitter": "", '
              '"folder": "fiscal/impots directs"}\n```')
    assert (p.name, p.folder) == ("2024 Bulletin d'imposition", "Fiscal/Impôts directs")


def test_invalid_fields_become_empty_never_guessed():
    p = parse('{"period": "March 2025", "nature": "Invoice", "emitter": "X", "folder": "Purchases"}')
    assert p.period == NO_DATE and p.nature == "" and p.folder == ""


def test_garbage():
    assert parse("I cannot read this document").nature == ""
    assert parse("{not json}").period == ""


def test_undated_yearly_folder_goes_to_sort():
    p = parse('{"period": "SANS-DATE", "nature": "Note de frais", "emitter": "", "folder": "Notes de frais"}')
    assert p.folder == TO_SORT


def test_prompt_never_contains_the_expected_answer():
    piece = {"company": "C", "filename": "f.pdf", "text": "T",
             "expected": {"period": "2031-01", "nature": "Facture", "emitter": "ZZSECRET", "folder": "x"}}
    flat = str(messages(piece))
    assert "ZZSECRET" not in flat and "2031-01" not in flat
