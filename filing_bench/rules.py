"""The rules engine, in four versions. Each version is a subclass of the
previous one and changes ONE idea, so the results table shows what each idea
is worth on its own.

  v0  naive        first date in "dd/mm/yyyy" (file name included), French
                   keywords in a fixed order, no emitter. When it finds no
                   date it uses the month the batch was received - the
                   classic "default to today" bug.
  v1  dates        every date format FR/EN/DE (numeric, words, 2-digit
                   years, quarters), weighed by the words just before it
                   ("due", "printed", "born" push away; "invoice date",
                   "period", "to" pull in); a range keeps its end; file name
                   ignored; no date found -> SANS-DATE, never a guess.
  v2  vocabulary   multilingual keywords weighed by zone (title lines count
                   more than the body, where a bank statement says
                   "invoice" and "salary"), and a period policy per nature
                   (a tax assessment is named by its tax year, a VAT return
                   by the end of its quarter, a contract by its signature).
  v3  parties      who issued the piece: the emitter from the letterhead
                   (legal form stripped), the customer of a sales invoice,
                   the client company recognised as issuer, a negative total
                   turning an "invoice" into a credit note, and the sender of
                   a letter choosing its folder. v3 also says how sure it is.
"""
import datetime as dt
import re

from . import dates
from .schema import NATURE_FOLDER, NO_DATE, Proposal, fold, place

RECEIVED = dt.date(2026, 1, 15)   # the day the batch arrived (v0 falls back to it)


class V0:
    name = "rules-v0"

    KEYWORDS = [  # first hit wins
        ("relevé", "Relevé bancaire"), ("facture", "Facture"), ("avoir", "Note de crédit"),
        ("salaire", "Fiche de salaire"), ("cotisation", "Décompte cotisations sociales"),
        ("tva", "Déclaration TVA"), ("impôt", "Bulletin d'imposition"), ("statuts", "Statuts coordonnés"),
        ("procès-verbal", "Procès-verbal AG"), ("bail", "Contrat de bail"),
        ("contrat", "Contrat de prestation"), ("assurance", "Attestation d'assurance"),
    ]

    def propose(self, piece):
        nature = self.nature(piece)
        period = self.period(piece, nature)
        emitter = self.emitter(piece, nature)
        p = Proposal(period=period, nature=nature, emitter=emitter)
        p.folder = self.folder(piece, p)
        return p

    def nature(self, piece):
        low = piece["text"].lower()
        for word, nature in self.KEYWORDS:
            if word in low:
                return nature
        return "Courrier"

    def period(self, piece, nature):
        for source in (piece["filename"], piece["text"]):
            m = re.search(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)", source) if source is piece["filename"] else None
            if m and 1 <= int(m[2]) <= 12:
                return f"{m[1]}-{m[2]}"
            m = re.search(r"(\d{2})/(\d{2})/(\d{4})", source)
            if m:
                return f"{m[3]}-{m[2]}"
        return f"{RECEIVED.year}-{RECEIVED.month:02d}"

    def emitter(self, piece, nature):
        return ""

    def folder(self, piece, p):
        return place(NATURE_FOLDER.get(p.nature, "Courrier divers"), p.period)


class V1(V0):
    name = "rules-v1"

    # Words that, just before a date, tell whether it is THE date of the piece.
    CUES = [
        (r"date de facture|invoice date|rechnungsdatum|\bdate\s*:|\bdatum\s*:", 3),
        (r"periode|period|zeitraum|abrechnungsmonat|mois de paie|\bmonth\b|\bmonat\b", 2),
        (r"\bau\s*$|\bto\s*$|\bbis\s*$", 3),           # end of a range
        (r"tenue le|held on|\bvom\s*$|\bdu\s*$|\bof\s*$|\bam\s*$", 1),
        (r"fait a .*le\s*$|signed in .*on\s*$|luxemburg, den\s*$|luxembourg, le\s*$", 2),
        (r"statuts coordonnes au|as at|stand", 2),
        (r"echeance|due|fallig|zahlbar|a payer|payable|pour le\s*$", -4),
        (r"imprime|printed|gedruckt", -4),
        (r"ne\(e\) le|birth|geburt|entree|start date|eintritt", -5),
        (r"paye le|paid on|ausgezahlt", -2),
        (r"commande|order|bestellung|depuis|since|seit", -4),
        (r"constituee|incorporated|gegrundet|notarie", -4),
        (r"clos le|ended|zum\s*$|convocation|notice|einladung", -3),
        (r"prend effet|commences|beginnt|jusqu|ends on|endet", -3),
        (r"transmise|filed on|ubermittelt|etabli le|erstellt am", -2),
        (r"n[°o]\.?\s*$|nr\.?\s*$|no\.\s*$|ticket|police|policy|iban|tva lu|r\.c\.s", -6),
    ]

    def score(self, mention):
        s = 0.0
        for rx, w in self.CUES:
            if re.search(rx, mention.before):
                s += w
        s += {"full": 0.5, "month": 0.5, "quarter": 0.5, "year": -3}[mention.kind]
        return s

    def candidates(self, piece):
        return [m for m in dates.find(piece["text"]) if m.kind != "year"]

    def period(self, piece, nature):
        cands = self.candidates(piece)
        if not cands:
            return NO_DATE
        best = max(cands, key=lambda m: (self.score(m), -m.start))
        return best.period


class V2(V1):
    name = "rules-v2"

    # (nature, pattern on plain text, weight). Most specific patterns carry more.
    VOCAB = [
        ("Relevé bancaire", r"releve de compte|extrait de compte|account statement|statement of account|kontoauszug|\biban\b", 3),
        ("Facture", r"\bfacture\b|\binvoice\b|\brechnung\b", 2),
        ("Note de crédit", r"\bavoir\b|note de credit|credit note|credit memo|gutschrift", 4),
        ("Note de frais", r"\bticket\b|total eur|merci de votre visite|thank you|vielen dank", 2),
        ("Fiche de salaire", r"fiche de salaire|bulletin de paie|payslip|salary statement|lohnabrechnung|gehaltsabrechnung|net a payer|net pay|auszahlung", 3),
        ("Décompte cotisations sociales", r"cotisations|contribution statement|statement of contributions|beitrag", 3),
        ("Déclaration TVA", r"declaration de tva|declaration tva|vat return|vat declaration|umsatzsteuer|mwst-erklarung|declaration annuelle de tva", 4),
        ("Bulletin d'imposition", r"bulletin de l.impot|bulletin d.etablissement|tax assessment|steuerbescheid|korperschaftsteuerbescheid", 4),
        ("Statuts coordonnés", r"statuts coordonnes|articles of association|consolidated articles|koordinierte (statuten|satzung)", 4),
        ("Procès-verbal AG", r"proces-verbal|assemblee generale|general meeting|generalversammlung|gesellschafterversammlung|minutes", 4),
        ("Contrat de bail", r"\bbail\b|\blease\b|mietvertrag|\bloyer\b", 4),
        ("Contrat de prestation", r"prestation de services|contrat de services|service agreement|services contract|dienstleistungsvertrag|dienstvertrag", 4),
        ("Contrat de travail", r"contrat de travail|employment contract|contract of employment|arbeitsvertrag", 4),
        ("Attestation d'assurance", r"attestation d.assurance|certificat d.assurance|certificate of insurance|insurance certificate|versicherungsbestatigung|versicherungsnachweis", 4),
        ("Courrier", r"\bconcerne\b|\bsubject\b|\bbetreff\b", 3),
    ]
    TITLE_LINES = 6
    ZONE = (3.0, 0.5)     # weight of a hit in the title lines, in the body

    def nature_scores(self, piece):
        lines = dates.plain(piece["text"]).split("\n")
        title, body = "\n".join(lines[:self.TITLE_LINES]), "\n".join(lines[self.TITLE_LINES:])
        scores = {}
        for nature, rx, w in self.VOCAB:
            s = 0.0
            if re.search(rx, title):
                s += w * self.ZONE[0]
            if re.search(rx, body):
                s += w * self.ZONE[1]
            if s:
                scores[nature] = scores.get(nature, 0) + s
        return scores

    def nature(self, piece):
        scores = self.nature_scores(piece)
        if not scores:
            return "Courrier"
        return max(scores, key=lambda n: (scores[n], -list(dict.fromkeys(v[0] for v in self.VOCAB)).index(n)))

    # The period policy: which date names the piece depends on what the piece is.
    def period(self, piece, nature):
        ms = dates.find(piece["text"])
        if nature == "Bulletin d'imposition":
            for m in ms:
                if m.kind == "year" and re.search(r"impot|tax|steuer|imposition|annee|year|jahr|collectivites", m.before + " "):
                    return m.period
            years = [m for m in ms if m.kind == "year"]
            return years[0].period if years else NO_DATE
        if nature == "Déclaration TVA":
            for m in ms:
                if m.kind == "quarter":
                    return m.period
            if re.search(r"annuelle|annual|jahres", dates.plain(piece["text"])):
                years = [m for m in ms if m.kind == "year" and re.search(r"exercice|year|jahr", m.before)]
                if years:
                    return years[0].period
        if nature in ("Fiche de salaire", "Décompte cotisations sociales", "Déclaration TVA"):
            months = [m for m in ms if m.kind == "month"]
            if months:
                return months[0].period
        if nature.startswith("Contrat"):
            signed = [m for m in ms if re.search(r"fait a|signed|, den\s*$|, le\s*$", m.before)]
            return signed[-1].period if signed else NO_DATE
        if nature == "Attestation d'assurance":
            issued = [m for m in ms if re.search(r"delivree|issued|ausgestellt", m.before)]
            if issued:
                return issued[0].period
        if nature in ("Statuts coordonnés", "Procès-verbal AG"):
            heads = [m for m in ms if m.kind == "full" and m.line <= 2]
            if heads:
                return heads[0].period
        return super().period(piece, nature)


LEGAL_FORMS = r"\b(s\.?\s?a\.?\s?r\.?\s?l\.?(-s)?|s\.?\s?a\.?|s\.?\s?c\.?|sarl(-s)?|gmbh|ag|ltd\.?|plc|se|sci|s\.?\s?à\s?r\.?\s?l\.?)\s*$"
SMALL = {"de", "du", "des", "la", "le", "les", "et", "d", "l"}


def short_name(raw):
    """'PAPETERIE WEBER S.À R.L.' -> 'Papeterie Weber'."""
    name = raw.split(" / ")[0].strip(" :.,")
    for _ in range(2):
        name = re.sub(LEGAL_FORMS, "", name, flags=re.I).strip(" ,")
    if name.isupper():
        words = name.lower().split()
        name = " ".join(w if i and w in SMALL else w[:1].upper() + w[1:] for i, w in enumerate(words))
    return name


class V3(V2):
    name = "rules-v3"

    def header_line(self, piece):
        """The letterhead. When it is only a logo, a web or mail address, the
        name survives in the footer, recognisable by its legal form."""
        lines = [l.strip() for l in piece["text"].split("\n") if l.strip()]
        if not lines:
            return ""
        if re.search(r"www\.|@|\[logo\]", lines[0]):
            for line in reversed(lines):
                head = line.split(" - ")[0]
                if re.search(LEGAL_FORMS, head, re.I):
                    return head
        return lines[0]

    def issued_by_client(self, piece):
        return fold(piece["company"]) in fold(self.header_line(piece))

    def negative_total(self, piece):
        return bool(re.search(r"(total|gesamt|subtotal|netto)[^\n]*:\s*-\s?\d", piece["text"], re.I))

    def nature(self, piece):
        n = super().nature(piece)
        if n in ("Facture", "Note de crédit", "Courrier") and self.issued_by_client(piece) \
                and re.search(r"factur|invoice|rechnung|honorar|fee note", dates.plain(piece["text"])):
            return "Facture client"
        if n in ("Facture", "Courrier") and self.negative_total(piece):
            return "Note de crédit"
        return n

    def emitter(self, piece, nature):
        if nature == "Facture client":
            m = re.search(r"(?:facture a|bill to|rechnung an)\s*:\s*(.+)", dates.plain(piece["text"]))
            return short_name(piece["text"][m.start(1):m.end(1)]) if m else ""
        if nature in ("Relevé bancaire", "Facture", "Note de crédit", "Attestation d'assurance", "Courrier"):
            return short_name(self.header_line(piece))
        return ""

    SENDER_FOLDER = [(r"fiscal|steuer|tax", "Fiscal/Impôts directs"), (r"\btva\b|vat", "TVA"),
                     (r"cotisations|social|sozial", "Cotisations sociales"),
                     (r"banque|bank|caisse", "Banque")]

    def folder(self, piece, p):
        if p.nature == "Courrier":
            sender = dates.plain(p.emitter)
            for rx, base in self.SENDER_FOLDER:
                if re.search(rx, sender):
                    return place(base, p.period)
        return super().folder(piece, p)

    def propose(self, piece):
        p = super().propose(piece)
        p.confidence = self.confidence(piece, p)
        return p

    def confidence(self, piece, p):
        """How sure the rules are, from 0 to 1. The hybrid asks a model below a threshold."""
        scores = sorted(self.nature_scores(piece).values(), reverse=True)
        c = 1.0
        if not scores:
            c -= 0.5
        elif len(scores) > 1 and scores[0] - scores[1] < 3:
            c -= 0.3
        if p.period == NO_DATE:
            c -= 0.3
        if p.nature in ("Facture", "Note de crédit", "Relevé bancaire", "Courrier") and len(p.emitter) < 3:
            c -= 0.3
        return max(c, 0.0)


VERSIONS = [V0(), V1(), V2(), V3()]
