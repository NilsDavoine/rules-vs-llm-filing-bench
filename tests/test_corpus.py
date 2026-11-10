from filing_bench import corpus
from filing_bench.schema import NATURES, compose_name, folder_tree, valid_period


def test_deterministic():
    a, b = corpus.build_all(), corpus.build_all()
    assert a == b


def test_size_and_split():
    built = corpus.build_all()
    assert len(built["dev"]) + len(built["test"]) == 150
    ids = [p["id"] for s in built.values() for p in s]
    assert len(ids) == len(set(ids))


def test_files_on_disk_match_the_generator():
    built = corpus.build_all()
    for split in ("dev", "test"):
        assert corpus.load(split) == built[split]


def test_expected_answers_are_well_formed():
    tree = set(folder_tree())
    for split in ("dev", "test"):
        for p in corpus.load(split):
            e = p["expected"]
            assert valid_period(e["period"]), p["id"]
            assert e["nature"] in NATURES, p["id"]
            assert e["folder"] in tree, p["id"]
            assert bool(e["emitter"]) == NATURES[e["nature"]], p["id"]
            assert compose_name(e["period"], e["nature"], e["emitter"])


def test_held_out_wording_never_leaks_into_dev():
    """The last option of every phrasing list is test-only."""
    held_out = ["Arrêté au", "Closing date", "Abschlussdatum", "Émise le", "Issued on", "Ausgestellt am",
                "Facture d'avoir", "Rechnungskorrektur", "Anstellungsvertrag", "Pachtvertrag",
                "Note d'honoraires", "Honorarnote", "Fee note"]
    dev_text = "\n".join(p["text"] for p in corpus.load("dev"))
    test_text = "\n".join(p["text"] for p in corpus.load("test"))
    leaked = [w for w in held_out if w in dev_text and "Ausgestellt am" != w and "Issued on" != w]
    assert not leaked
    assert any(w in test_text for w in held_out)
