"""The filing convention every approach must follow.

A proposal is four fields: period, nature, emitter, folder. The file name is
built from them by `compose_name`, never written freely, so rules and models
are judged on exactly the same thing.

    name   = "<period> <nature>[ <emitter>]"
    period = "YYYY-MM" | "YYYY" | "SANS-DATE"   (SANS-DATE: "no date" in French)

The firm names its files in French whatever the language of the document.
"""
from dataclasses import dataclass, asdict
import re
import unicodedata

NO_DATE = "SANS-DATE"

# Closed list of natures. True = the emitter is part of the name, because it
# tells two pieces of the same nature apart ("Facture Papeterie Weber").
NATURES = {
    "Relevé bancaire": True,
    "Facture": True,
    "Note de crédit": True,
    "Facture client": True,          # issued BY the client company, named after its customer
    "Note de frais": False,
    "Fiche de salaire": False,
    "Décompte cotisations sociales": False,
    "Déclaration TVA": False,
    "Bulletin d'imposition": False,
    "Statuts coordonnés": False,
    "Procès-verbal AG": False,
    "Contrat de bail": False,
    "Contrat de prestation": False,
    "Contrat de travail": False,
    "Attestation d'assurance": True,
    "Courrier": True,
}

# Folders that hold the year of the period: "2025/Banque".
YEARLY = ["Banque", "Achats", "Ventes", "Notes de frais", "Salaires",
          "Cotisations sociales", "TVA"]
PERMANENT = ["Fiscal/Impôts directs", "Juridique/Statuts", "Juridique/Assemblées",
             "Juridique/Contrats", "Assurances", "Courrier divers"]
TO_SORT = "À classer"      # a yearly piece with no date cannot pick its year
YEARS = ["2022", "2023", "2024", "2025", "2026"]


def folder_tree():
    return [f"{y}/{f}" for y in YEARS for f in YEARLY] + PERMANENT + [TO_SORT]


NATURE_FOLDER = {
    "Relevé bancaire": "Banque", "Facture": "Achats", "Note de crédit": "Achats",
    "Facture client": "Ventes", "Note de frais": "Notes de frais",
    "Fiche de salaire": "Salaires", "Décompte cotisations sociales": "Cotisations sociales",
    "Déclaration TVA": "TVA", "Bulletin d'imposition": "Fiscal/Impôts directs",
    "Statuts coordonnés": "Juridique/Statuts", "Procès-verbal AG": "Juridique/Assemblées",
    "Contrat de bail": "Juridique/Contrats", "Contrat de prestation": "Juridique/Contrats",
    "Contrat de travail": "Juridique/Contrats", "Attestation d'assurance": "Assurances",
    "Courrier": "Courrier divers",
}


def place(base_folder, period):
    """Turn a base folder into a path of the tree, using the period's year."""
    if base_folder not in YEARLY:
        return base_folder
    if period == NO_DATE or not period:
        return TO_SORT
    return f"{period[:4]}/{base_folder}"


@dataclass
class Proposal:
    period: str = ""
    nature: str = ""
    emitter: str = ""
    folder: str = ""
    confidence: float = 1.0      # used by the hybrid approach only
    raw: str = ""                # what a model actually answered, for error analysis

    @property
    def name(self):
        return compose_name(self.period, self.nature, self.emitter)

    def to_dict(self):
        d = asdict(self)
        d["name"] = self.name
        return d


def compose_name(period, nature, emitter=""):
    parts = [period or NO_DATE, nature or "Document"]
    if emitter and NATURES.get(nature, False):
        parts.append(emitter)
    return clean(" ".join(parts))


_FORBIDDEN = re.compile(r'["*:<>?/\\|\x00-\x1f]')


def clean(text):
    return re.sub(r"\s+", " ", _FORBIDDEN.sub(" ", text)).strip()


def fold(text):
    """Comparison key: no accents, no case, single spaces, no punctuation."""
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(c for c in text if unicodedata.category(c) != "Mn").lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


PERIOD_RE = re.compile(r"^(\d{4})(?:-(0[1-9]|1[0-2]))?$")


def valid_period(period):
    return period == NO_DATE or bool(PERIOD_RE.match(period or ""))
