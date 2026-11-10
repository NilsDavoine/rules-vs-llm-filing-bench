"""A deterministic corpus of invented accounting pieces.

Every piece is what a filing assistant really receives: the text extracted
from a PDF or a photo, a messy original file name, and the client company it
belongs to. Each one carries the answer a careful accountant would give.

Nothing here comes from a real client: companies, suppliers, banks, amounts,
identifiers and addresses are generated.

Two splits, built so that the test split measures generalisation, not memory:
  * dev  (99 pieces, seed 11): emitter pool A, phrasing variants 1..n-1.
    The rules were written while looking at this split only.
  * test (51 pieces, seed 29): emitter pool B, and every list of phrasings may
    use its LAST option, which never appears in dev ("held-out" wording).

    python -m filing_bench.corpus      # rewrites data/dev.jsonl and data/test.jsonl
"""
import datetime as dt
import json
import os
import random
import re

from .schema import NATURE_FOLDER, NO_DATE, place

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

MONTHS = {
    "fr": ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
           "septembre", "octobre", "novembre", "décembre"],
    "en": ["January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December"],
    "de": ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August",
           "September", "Oktober", "November", "Dezember"],
}

# --------------------------------------------------------------- the pools --
# (short name used in the expected file name, legal form shown on the document, language)
POOLS = {
    "A": {
        "companies": [("Brasserie du Plateau", "S.à r.l."), ("Nordlicht Consulting", "S.A.")],
        "banks": [("Banque Atlas", "S.A."), ("Caisse Mosane", "S.C."), ("Kapellen Bank", "AG")],
        "suppliers": [("Papeterie Weber", "S.à r.l.", "fr"), ("Garage du Viaduc", "S.à r.l.", "fr"),
                      ("Électricité Thill", "SARL-S", "fr"), ("Imprimerie Rousseau", "S.A.", "fr"),
                      ("Brightdesk Software", "Ltd", "en"), ("Cloudvale Hosting", "Ltd", "en"),
                      ("Bürotechnik Kremer", "GmbH", "de"), ("Schreinerei Hoffmann", "GmbH", "de"),
                      ("Netzwerk Mosel", "AG", "de")],
        "customers": [("Boulangerie des Remparts", "S.à r.l."), ("Hôtel Belvédère", "S.A."),
                      ("Westbridge Partners", "Ltd"), ("Kanzlei Reuter", "GmbH")],
        "insurers": [("Assurances Lionel", "S.A."), ("Mutuelle Ardennaise", "S.A.")],
        "landlords": [("Immobilière Glacis", "S.A.")],
        "merchants": ["Restaurant Le Quai", "Station Express Pétange", "Parking Centre-Gare",
                      "Hotel Sonnenhof", "Café des Arts"],
    },
    "B": {
        "companies": [("Atelier Vallon", "SARL-S"), ("Nordlicht Consulting", "S.A.")],
        "banks": [("Banque Hélios", "S.A."), ("Nordwand Privatbank", "AG")],
        "suppliers": [("Traiteur Bastin", "S.à r.l.", "fr"), ("Nettoyage Clément", "S.A.", "fr"),
                      ("Ledgerline Analytics", "Ltd", "en"), ("Harbourview Couriers", "Ltd", "en"),
                      ("Malerbetrieb Scholtes", "GmbH", "de"), ("Kaffeerösterei Wagner", "GmbH", "de")],
        "customers": [("Atelier Lumen", "S.à r.l."), ("Domaine des Coteaux", "S.A."),
                      ("Fernhill Studio", "Ltd")],
        "insurers": [("Rhein-Mosel Versicherung", "AG")],
        "landlords": [("SCI Verger", "")],
        "merchants": ["Brasserie Gare-Nord", "Tankstelle Wasserbillig", "Taxi Kirchberg"],
    },
}

TAX_OFFICE = "Administration fiscale"
VAT_OFFICE = "Office de la TVA"
SOCIAL_OFFICE = "Office des cotisations sociales"

# How many pieces of each kind, for 150 in total.
MIX = [("bank_statement", 18), ("supplier_invoice", 26), ("credit_note", 8),
       ("sales_invoice", 12), ("receipt", 12), ("payslip", 14), ("social", 8), ("vat", 10),
       ("tax_assessment", 8), ("articles", 4), ("minutes", 6), ("contract", 10),
       ("insurance", 6), ("letter", 8)]


class Gen:
    def __init__(self, split, seed, pool):
        self.split = split
        self.rng = random.Random(seed)
        self.pool = POOLS[pool]

    # A list of phrasings: dev never sees the last one.
    def pick(self, options):
        if self.split == "dev" and len(options) > 1:
            return self.rng.choice(options[:-1])
        return self.rng.choice(options)

    def chance(self, p):
        return self.rng.random() < p

    def date(self, year_from=2023, year_to=2025):
        y = self.rng.randint(year_from, year_to)
        m = self.rng.randint(1, 12)
        return dt.date(y, m, self.rng.randint(1, 28))

    # ------------------------------------------------------- date writing --
    def fmt(self, d, lang, style=None):
        style = style or self.pick(["slash", "dot", "long", "iso"])
        if style == "slash":
            return d.strftime("%d/%m/%Y")
        if style == "dot":
            return d.strftime("%d.%m.%Y")
        if style == "iso":
            return d.isoformat()
        month = MONTHS[lang][d.month - 1]
        if lang == "fr":
            return f"{'1er' if d.day == 1 else d.day} {month} {d.year}"
        if lang == "de":
            return f"{d.day}. {month} {d.year}"
        if self.chance(0.5):
            return f"{d.day} {month} {d.year}"
        return f"{month} {d.day}, {d.year}"

    def fmt_month(self, d, lang):
        if self.chance(0.6):
            return f"{MONTHS[lang][d.month - 1]} {d.year}"
        return d.strftime(self.pick(["%m/%Y", "%m.%Y"]))

    def money(self, lo, hi):
        v = self.rng.uniform(lo, hi)
        return f"{v:,.2f}".replace(",", " ").replace(".", ",")

    def header(self, name, form):
        full = f"{name} {form}".strip()
        return full.upper() if self.chance(0.5) else full

    def address(self):
        streets = ["rue des Tanneurs", "route d'Arlon", "boulevard des Tilleuls", "Hauptstrooss",
                   "rue du Verger", "avenue de la Faïencerie", "Grand-Rue", "rue de la Gare"]
        towns = ["Luxembourg", "Esch-sur-Alzette", "Differdange", "Ettelbruck", "Mersch",
                 "Strassen", "Bertrange", "Remich"]
        return (f"{self.rng.randint(1, 120)}, {self.rng.choice(streets)} · "
                f"L-{self.rng.randint(1000, 9999)} {self.rng.choice(towns)}")

    def vat_id(self):
        return f"LU{self.rng.randint(10000000, 99999999)}"


def _end_of_month(d):
    nxt = (d.replace(day=28) + dt.timedelta(days=4))
    return nxt - dt.timedelta(days=nxt.day)


def _shift_month(d, n):
    m = d.month - 1 + n
    return dt.date(d.year + m // 12, m % 12 + 1, min(d.day, 28))


def _period(d):
    return f"{d.year}-{d.month:02d}"


# ------------------------------------------------------------ the builders --
# Each returns (lines, period, nature, emitter, base_folder, traps).

def bank_statement(g, lang, company, _):
    bank, form = g.rng.choice(g.pool["banks"])
    start = g.date().replace(day=1)
    end = _end_of_month(start)
    printed = end + dt.timedelta(days=g.rng.randint(1, 6))
    number = f"{g.rng.randint(1, 12)}/{end.year}"
    title = {"fr": ["RELEVÉ DE COMPTE", "Extrait de compte", "Relevé de compte courant"],
             "en": ["ACCOUNT STATEMENT", "Statement of account", "Bank statement"],
             "de": ["KONTOAUSZUG", "Kontoauszug Nr.", "Umsatzübersicht"]}[lang]
    a, b = g.fmt(start, lang, "slash" if lang != "de" else "dot"), None
    b = g.fmt(end, lang, "slash" if lang != "de" else "dot")
    period_line = {"fr": [f"Période du {a} au {b}", f"Du {a} au {b}", f"Arrêté au {b}"],
                   "en": [f"Statement period: {a} to {b}", f"From {a} to {b}", f"Closing date: {b}"],
                   "de": [f"Zeitraum: {a} bis {b}", f"vom {a} bis {b}", f"Abschlussdatum: {b}"]}[lang]
    labels = {"fr": ("Titulaire", "Imprimé le", "Solde au", "Frais de tenue de compte", "Salaire"),
              "en": ("Account holder", "Printed on", "Balance at", "Account fees", "Salary"),
              "de": ("Kontoinhaber", "Gedruckt am", "Saldo per", "Kontoführungsgebühr", "Gehalt")}[lang]
    sup = g.rng.choice(g.pool["suppliers"])[0]
    lines = [g.header(bank, form), g.address(), f"{g.pick(title)} {number}",
             f"{labels[0]} : {company}", f"IBAN LU{g.rng.randint(10**17, 10**18 - 1)}",
             g.pick(period_line), ""]
    prev = MONTHS[lang][_shift_month(start, -1).month - 1]
    tx = [f"VIR SEPA {sup.upper()} {'Facture' if lang == 'fr' else 'Invoice' if lang == 'en' else 'Rechnung'} {g.rng.randint(100, 999)}",
          f"{labels[4]} {prev}", labels[3], f"DOM {g.rng.choice(g.pool['insurers'])[0].upper()}",
          f"VIR {g.rng.choice(g.pool['customers'])[0].upper()}"]
    for label in g.rng.sample(tx, k=g.rng.randint(3, 5)):
        day = start + dt.timedelta(days=g.rng.randint(0, 26))
        lines.append(f"{day.strftime('%d.%m')}  {label:<42} {g.money(-4000, 6000)}")
    lines += ["", f"{labels[2]} {b} : {g.money(1000, 90000)} EUR", f"{labels[1]} {g.fmt(printed, lang, 'slash')}"]
    traps = ["several_dates", "print_date_next_month", "number_looks_like_month", "body_keywords"]
    return lines, _period(end), "Relevé bancaire", bank, "Banque", traps


def _invoice_dates(g, lang, d, lines_after_client=None):
    due = d + dt.timedelta(days=g.rng.choice([15, 30, 45]))
    date_lab = {"fr": ["Date de facture", "Date", "Émise le"],
                "en": ["Invoice date", "Date", "Issued on"],
                "de": ["Rechnungsdatum", "Datum", "Ausgestellt am"]}[lang]
    due_lab = {"fr": ["Échéance", "À payer avant le", "Date limite de paiement"],
               "en": ["Due date", "Payment due", "Pay by"],
               "de": ["Fällig am", "Zahlbar bis", "Zahlungsziel"]}[lang]
    issued = f"{g.pick(date_lab)} : {g.fmt(d, lang)}"
    due_line = f"{g.pick(due_lab)} : {g.fmt(due, lang)}"
    traps = ["due_date"]
    if due.month != d.month:
        traps.append("due_date_other_month")
    if g.chance(0.35):
        traps.append("due_before_issue_in_layout")
        return [due_line, issued], traps
    return [issued, due_line], traps


def _items(g, lang, n=None):
    words = {"fr": ["Ramettes papier A4", "Entretien véhicule", "Main d'œuvre", "Licence annuelle",
                    "Déplacement", "Cartouches d'encre", "Installation électrique"],
             "en": ["Annual licence", "Hosting plan", "Support hours", "Domain renewal", "Setup fee"],
             "de": ["Wartung", "Büromaterial", "Arbeitsstunden", "Material", "Anfahrt"]}[lang]
    return [f"{w:<34} {g.rng.randint(1, 12):>3} x {g.money(5, 400)}"
            for w in g.rng.sample(words, k=n or g.rng.randint(1, 3))]


def _totals(g, lang, sign=""):
    lab = {"fr": ("Total HTVA", "TVA 17 %", "Total TTC"), "en": ("Subtotal", "VAT 17%", "Total"),
           "de": ("Netto", "MwSt. 17 %", "Gesamtbetrag")}[lang]
    net = round(g.rng.uniform(50, 3000), 2)
    tax = round(net * 0.17, 2)

    def eur(v):
        return f"{sign}{v:,.2f}".replace(",", " ").replace(".", ",")
    return [f"{lab[0]} : {eur(net)} EUR", f"{lab[1]} : {eur(tax)} EUR", f"{lab[2]} : {eur(net + tax)} EUR"]


def supplier_invoice(g, lang, company, undated):
    candidates = [s for s in g.pool["suppliers"] if s[2] == lang] or g.pool["suppliers"]
    sup, form, _ = g.rng.choice(candidates)
    d = g.date()
    title = {"fr": ["FACTURE", "Facture n°", "FACTURE ORIGINALE"], "en": ["INVOICE", "Tax invoice", "Invoice no."],
             "de": ["RECHNUNG", "Rechnung Nr.", "Rechnung (Original)"]}[lang]
    client = {"fr": "Client", "en": "Bill to", "de": "Kunde"}[lang]
    lines = [g.header(sup, form), g.address(), f"TVA {g.vat_id()}", "",
             f"{g.pick(title)} {d.year % 100:02d}-{g.rng.randint(1000, 9999)}",
             f"{client} : {company}", g.address()]
    traps = []
    if undated:
        traps.append("date_cut_off")
    else:
        dates, traps = _invoice_dates(g, lang, d)
        lines += dates
        if g.chance(0.3):
            order = d - dt.timedelta(days=g.rng.randint(20, 60))
            lab = {"fr": "Votre commande du", "en": "Your order of", "de": "Ihre Bestellung vom"}[lang]
            lines.append(f"{lab} {g.fmt(order, lang)}")
            traps.append("order_date")
        if g.chance(0.25):
            lab = {"fr": "Client depuis", "en": "Customer since", "de": "Kunde seit"}[lang]
            lines.append(f"{lab} {g.rng.randint(2012, 2021)}")
            traps.append("customer_since_year")
    lines += [""] + _items(g, lang) + [""] + _totals(g, lang)
    pay = {"fr": "Merci de virer le montant sur notre compte en mentionnant le numéro de facture.",
           "en": "Please transfer the amount quoting the invoice number.",
           "de": "Bitte überweisen Sie den Betrag unter Angabe der Rechnungsnummer."}[lang]
    lines.append(pay)
    period = NO_DATE if undated else _period(d)
    return lines, period, "Facture", sup, "Achats", traps


def credit_note(g, lang, company, _):
    candidates = [s for s in g.pool["suppliers"] if s[2] == lang] or g.pool["suppliers"]
    sup, form, _ = g.rng.choice(candidates)
    d = g.date()
    orig = d - dt.timedelta(days=g.rng.randint(25, 70))
    title = {"fr": ["AVOIR", "NOTE DE CRÉDIT", "Facture d'avoir"], "en": ["CREDIT NOTE", "Credit memo", "Credit invoice"],
             "de": ["GUTSCHRIFT", "Gutschrift Nr.", "Rechnungskorrektur"]}[lang]
    ref = {"fr": f"Concerne notre facture n° {g.rng.randint(1000, 9999)} du {g.fmt(orig, lang)}",
           "en": f"Relates to our invoice no. {g.rng.randint(1000, 9999)} dated {g.fmt(orig, lang)}",
           "de": f"Bezug: unsere Rechnung Nr. {g.rng.randint(1000, 9999)} vom {g.fmt(orig, lang)}"}[lang]
    date_lab = {"fr": "Date", "en": "Date", "de": "Datum"}[lang]
    reason = {"fr": "Retour de marchandise", "en": "Returned goods", "de": "Warenrücksendung"}[lang]
    lines = [g.header(sup, form), g.address(), "", f"{g.pick(title)} {g.rng.randint(10, 99)}",
             f"{date_lab} : {g.fmt(d, lang)}", f"{company}", ref, "", reason] + _totals(g, lang, "-")
    return lines, _period(d), "Note de crédit", sup, "Achats", ["mentions_invoice", "original_invoice_date"]


def sales_invoice(g, lang, company, undated):
    form = dict(g.pool["companies"])[company]
    cust, cform = g.rng.choice(g.pool["customers"])
    d = g.date()
    title = {"fr": ["FACTURE", "Facture n°", "Note d'honoraires"], "en": ["INVOICE", "Invoice no.", "Fee note"],
             "de": ["RECHNUNG", "Rechnung Nr.", "Honorarnote"]}[lang]
    to = {"fr": "Facturé à", "en": "Bill to", "de": "Rechnung an"}[lang]
    lines = [g.header(company, form), g.address(), f"TVA {g.vat_id()}", "",
             f"{g.pick(title)} {d.year}-{g.rng.randint(10, 300):03d}",
             f"{to} : {g.header(cust, cform)}", g.address()]
    traps = ["issuer_is_the_client_company"]
    if undated:
        traps.append("date_cut_off")
    else:
        dates, more = _invoice_dates(g, lang, d)
        lines += dates
        traps += more
    lines += [""] + _items(g, lang) + [""] + _totals(g, lang)
    period = NO_DATE if undated else _period(d)
    return lines, period, "Facture client", cust, "Ventes", traps


def receipt(g, lang, company, undated):
    merchant = g.rng.choice(g.pool["merchants"])
    d = g.date()
    t = f"{g.rng.randint(7, 22):02d}:{g.rng.randint(0, 59):02d}"
    traps = ["short_text"]
    style = g.rng.choice(["short", "dot", "slash"])
    if style == "short":
        when = f"{d.strftime('%d.%m.%y')} {t}"
        traps.append("two_digit_year")
    else:
        when = f"{g.fmt(d, lang, style)} {t}"
    thanks = {"fr": "Merci de votre visite", "en": "Thank you", "de": "Vielen Dank"}[lang]
    items = {"fr": ["Plat du jour", "Café", "Eau minérale"], "en": ["Lunch menu", "Coffee"],
             "de": ["Super 95", "Parkgebühr", "Mittagsmenü"]}[lang]
    lines = [merchant.upper(), g.address(), f"Ticket {g.rng.randint(1000, 99999)}"]
    if undated:
        traps.append("date_torn_off")
    else:
        lines.append(when)
    lines += [f"{i:<24}{g.money(2, 60)}" for i in g.rng.sample(items, k=min(2, len(items)))]
    lines += [f"TOTAL EUR {g.money(5, 150)}", f"TVA/MwSt {g.rng.choice(['3', '17'])}%", thanks]
    period = NO_DATE if undated else _period(d)
    return lines, period, "Note de frais", "", "Notes de frais", traps


def payslip(g, lang, company, _):
    form = dict(g.pool["companies"])[company]
    month = g.date().replace(day=1)
    end = _end_of_month(month)
    paid = end if g.chance(0.5) else end + dt.timedelta(days=g.rng.randint(1, 3))
    title = {"fr": ["FICHE DE SALAIRE", "Bulletin de paie", "Décompte de salaire"],
             "en": ["PAYSLIP", "Salary statement", "Pay advice"],
             "de": ["LOHNABRECHNUNG", "Gehaltsabrechnung", "Entgeltabrechnung"]}[lang]
    per = {"fr": ["Période", "Mois de paie", "Période de paie"], "en": ["Pay period", "Month", "Payroll month"],
           "de": ["Abrechnungsmonat", "Monat", "Abrechnungszeitraum"]}[lang]
    lab = {"fr": ("Salarié n°", "Né(e) le", "Date d'entrée", "Payé le", "Salaire de base",
                  "Cotisations sociales", "Impôt retenu", "Net à payer"),
           "en": ("Employee no.", "Date of birth", "Start date", "Paid on", "Base salary",
                  "Social contributions", "Tax withheld", "Net pay"),
           "de": ("Personalnr.", "Geburtsdatum", "Eintritt", "Ausgezahlt am", "Grundgehalt",
                  "Sozialbeiträge", "Lohnsteuer", "Auszahlung")}[lang]
    birth = dt.date(g.rng.randint(1960, 2002), g.rng.randint(1, 12), g.rng.randint(1, 28))
    entry = dt.date(g.rng.randint(2010, 2022), g.rng.randint(1, 12), 1)
    lines = [g.header(company, form), g.pick(title), f"{g.pick(per)} : {g.fmt_month(month, lang)}",
             f"{lab[0]} {g.rng.randint(1, 80):04d}", f"{lab[1]} {g.fmt(birth, lang, 'slash')}",
             f"{lab[2]} {g.fmt(entry, lang, 'slash')}", "",
             f"{lab[4]:<28}{g.money(2500, 7000)}", f"{lab[5]:<28}-{g.money(300, 900)}",
             f"{lab[6]:<28}-{g.money(100, 1500)}", f"{lab[7]:<28}{g.money(2000, 5000)}",
             f"{lab[3]} {g.fmt(paid, lang, 'slash')}"]
    traps = ["birth_date", "entry_date", "body_keywords"]
    if paid.month != month.month:
        traps.append("paid_next_month")
    return lines, _period(month), "Fiche de salaire", "", "Salaires", traps


def social(g, lang, company, _):
    month = g.date().replace(day=1)
    made = _shift_month(month, 1).replace(day=g.rng.randint(5, 12))
    due = _end_of_month(made)
    office = {"fr": SOCIAL_OFFICE, "en": f"{SOCIAL_OFFICE} / Social Contributions Office",
              "de": f"{SOCIAL_OFFICE} / Sozialbeitragsstelle"}[lang]
    title = {"fr": ["DÉCOMPTE DES COTISATIONS", "Extrait de compte cotisant", "Avis de débit des cotisations"],
             "en": ["STATEMENT OF CONTRIBUTIONS", "Contribution statement", "Contribution debit notice"],
             "de": ["BEITRAGSABRECHNUNG", "Beitragsnachweis", "Lastschriftanzeige Beiträge"]}[lang]
    lab = {"fr": ("Employeur", "Période de cotisation", "Établi le", "À payer pour le", "Montant dû"),
           "en": ("Employer", "Contribution period", "Issued on", "Payable by", "Amount due"),
           "de": ("Arbeitgeber", "Beitragszeitraum", "Erstellt am", "Zahlbar bis", "Fälliger Betrag")}[lang]
    lines = [office, g.address(), g.pick(title), f"{lab[0]} : {company}",
             f"{lab[1]} : {month.strftime('%m/%Y')}", f"{lab[2]} {g.fmt(made, lang)}",
             f"{lab[4]} : {g.money(800, 9000)} EUR", f"{lab[3]} {g.fmt(due, lang)}"]
    return lines, _period(month), "Décompte cotisations sociales", "", "Cotisations sociales", \
        ["issued_next_month", "due_date", "body_keywords"]


def vat(g, lang, company, _):
    form = dict(g.pool["companies"])[company]
    year = g.rng.randint(2023, 2025)
    kind = g.rng.choice(["quarter", "quarter", "month", "annual"])
    traps = []
    if kind == "quarter":
        q = g.rng.randint(1, 4)
        end = _end_of_month(dt.date(year, 3 * q, 1))
        ordinal = {1: "1er", 2: "2e", 3: "3e", 4: "4e"}[q]
        text = {"fr": [f"{ordinal} trimestre {year}", f"T{q} {year}",
                       f"Période : {dt.date(year, 3 * q - 2, 1).strftime('%d/%m/%Y')} - {end.strftime('%d/%m/%Y')}"],
                "en": [f"Q{q} {year}", f"Quarter {q} {year}",
                       f"Period {dt.date(year, 3 * q - 2, 1).strftime('%d/%m/%Y')} to {end.strftime('%d/%m/%Y')}"],
                "de": [f"{q}. Quartal {year}", f"Q{q}/{year}",
                       f"Zeitraum {dt.date(year, 3 * q - 2, 1).strftime('%d.%m.%Y')} - {end.strftime('%d.%m.%Y')}"]}[lang]
        when, period = g.pick(text), _period(end)
        traps.append("quarter_to_month")
        filed = _shift_month(end, 1).replace(day=g.rng.randint(5, 25))
    elif kind == "month":
        m = dt.date(year, g.rng.randint(1, 12), 1)
        when, period = g.fmt_month(m, lang), _period(m)
        filed = _shift_month(m, 1).replace(day=g.rng.randint(5, 15))
    else:
        when, period = {"fr": f"Exercice {year}", "en": f"Financial year {year}", "de": f"Geschäftsjahr {year}"}[lang], str(year)
        traps.append("year_only")
        filed = dt.date(year + 1, g.rng.randint(2, 4), g.rng.randint(1, 28))
    title = {"fr": ["DÉCLARATION DE TVA", "Déclaration TVA", "Relevé de TVA"],
             "en": ["VAT RETURN", "VAT declaration", "VAT filing"],
             "de": ["UMSATZSTEUERERKLÄRUNG", "MwSt-Erklärung", "Umsatzsteuer-Voranmeldung"]}[lang]
    if kind == "annual":
        title = {"fr": ["DÉCLARATION ANNUELLE DE TVA"], "en": ["ANNUAL VAT RETURN"],
                 "de": ["UMSATZSTEUERJAHRESERKLÄRUNG"]}[lang]
    lab = {"fr": ("Assujetti", "Transmise le", "TVA due"), "en": ("Taxable person", "Filed on", "VAT payable"),
           "de": ("Steuerpflichtiger", "Übermittelt am", "Zahllast")}[lang]
    lines = [VAT_OFFICE, g.pick(title), f"{lab[0]} : {g.header(company, form)} {g.vat_id()}",
             when, f"{lab[1]} {g.fmt(filed, lang, 'slash')}", f"{lab[2]} : {g.money(500, 20000)} EUR"]
    traps.append("filed_later")
    return lines, period, "Déclaration TVA", "", "TVA", traps


def tax_assessment(g, lang, company, _):
    year = g.rng.randint(2021, 2024)
    issued = dt.date(year + g.rng.randint(1, 2), g.rng.randint(1, 12), g.rng.randint(1, 28))
    office = TAX_OFFICE if lang == "fr" else f"{TAX_OFFICE} / " + ("Tax Administration" if lang == "en" else "Steuerverwaltung")
    title = {"fr": [f"BULLETIN DE L'IMPÔT SUR LE REVENU DES COLLECTIVITÉS {year}",
                    f"Bulletin d'établissement de l'impôt - année d'imposition {year}",
                    f"Avis d'imposition pour l'année {year}"],
             "en": [f"CORPORATE INCOME TAX ASSESSMENT {year}", f"Tax assessment notice - tax year {year}",
                    f"Notice of assessment for the year {year}"],
             "de": [f"KÖRPERSCHAFTSTEUERBESCHEID {year}", f"Steuerbescheid für das Veranlagungsjahr {year}",
                    f"Bescheid über die Körperschaftsteuer für {year}"]}[lang]
    lab = {"fr": ("Luxembourg, le", "Redevable", "Montant à payer dans le mois", "Bureau d'imposition Sociétés"),
           "en": ("Luxembourg,", "Taxpayer", "Amount payable within one month", "Corporate tax office"),
           "de": ("Luxemburg, den", "Steuerpflichtiger", "Innerhalb eines Monats zu zahlen", "Steueramt Körperschaften")}[lang]
    lines = [office, f"{lab[3]} {g.rng.randint(1, 9)}", f"{lab[0]} {g.fmt(issued, lang)}", "",
             g.pick(title), f"{lab[1]} : {company}", f"{lab[2]} : {g.money(500, 40000)} EUR"]
    return lines, str(year), "Bulletin d'imposition", "", "Fiscal/Impôts directs", \
        ["year_only", "issue_date_later_year", "letter_date"]


def articles(g, lang, company, _):
    form = dict(g.pool["companies"])[company]
    founded = dt.date(g.rng.randint(2008, 2018), g.rng.randint(1, 12), g.rng.randint(1, 28))
    d = g.date(2022, 2025)
    head = {"fr": [f"STATUTS COORDONNÉS au {g.fmt(d, lang, 'long')}", f"Statuts coordonnés suite à l'acte du {g.fmt(d, lang, 'long')}",
                   f"Texte coordonné des statuts en date du {g.fmt(d, lang, 'long')}"],
            "en": [f"COORDINATED ARTICLES OF ASSOCIATION as at {g.fmt(d, lang, 'long')}",
                   f"Consolidated articles following the deed of {g.fmt(d, lang, 'long')}",
                   f"Restated articles dated {g.fmt(d, lang, 'long')}"],
            "de": [f"KOORDINIERTE STATUTEN Stand: {g.fmt(d, lang, 'long')}",
                   f"Koordinierte Satzung nach der Urkunde vom {g.fmt(d, lang, 'long')}",
                   f"Neufassung der Satzung vom {g.fmt(d, lang, 'long')}"]}[lang]
    body = {"fr": f"La société a été constituée suivant acte notarié du {g.fmt(founded, lang, 'long')}. Art. 1er. Dénomination. Art. 2. Siège social.",
            "en": f"The company was incorporated by notarial deed of {g.fmt(founded, lang, 'long')}. Art. 1. Name. Art. 2. Registered office.",
            "de": f"Die Gesellschaft wurde durch notarielle Urkunde vom {g.fmt(founded, lang, 'long')} gegründet. Art. 1. Firma. Art. 2. Sitz."}[lang]
    lines = [g.header(company, form), f"R.C.S. Luxembourg B{g.rng.randint(100000, 299999)}", g.pick(head), body]
    return lines, _period(d), "Statuts coordonnés", "", "Juridique/Statuts", ["incorporation_date"]


def minutes(g, lang, company, _):
    form = dict(g.pool["companies"])[company]
    d = dt.date(g.rng.randint(2023, 2025), g.rng.choice([4, 5, 6]), g.rng.randint(1, 28))
    called = d - dt.timedelta(days=15)
    fy = dt.date(d.year - 1, 12, 31)
    head = {"fr": [f"PROCÈS-VERBAL DE L'ASSEMBLÉE GÉNÉRALE ORDINAIRE tenue le {g.fmt(d, lang, 'long')}",
                   f"Procès-verbal de l'assemblée générale annuelle du {g.fmt(d, lang, 'long')}",
                   f"Compte rendu de l'assemblée des associés réunie le {g.fmt(d, lang, 'long')}"],
            "en": [f"MINUTES OF THE ANNUAL GENERAL MEETING held on {g.fmt(d, lang, 'long')}",
                   f"Minutes of the ordinary general meeting of {g.fmt(d, lang, 'long')}",
                   f"Record of the shareholders' meeting convened on {g.fmt(d, lang, 'long')}"],
            "de": [f"PROTOKOLL DER ORDENTLICHEN GENERALVERSAMMLUNG vom {g.fmt(d, lang, 'long')}",
                   f"Niederschrift über die Gesellschafterversammlung am {g.fmt(d, lang, 'long')}",
                   f"Beschlussprotokoll der Jahresversammlung vom {g.fmt(d, lang, 'long')}"]}[lang]
    body = {"fr": [f"Convocation adressée le {g.fmt(called, lang)}.",
                   f"L'assemblée approuve les comptes de l'exercice clos le {g.fmt(fy, lang, 'long')}.",
                   "Décharge est donnée au gérant."],
            "en": [f"Notice sent on {g.fmt(called, lang)}.",
                   f"The meeting approves the accounts for the year ended {g.fmt(fy, lang, 'long')}.",
                   "Discharge is granted to the manager."],
            "de": [f"Einladung versandt am {g.fmt(called, lang)}.",
                   f"Die Versammlung genehmigt den Jahresabschluss zum {g.fmt(fy, lang, 'long')}.",
                   "Dem Geschäftsführer wird Entlastung erteilt."]}[lang]
    lines = [g.header(company, form), g.pick(head)] + body
    return lines, _period(d), "Procès-verbal AG", "", "Juridique/Assemblées", \
        ["financial_year_end", "notice_date"]


def contract(g, lang, company, undated):
    kind = g.rng.choice(["Contrat de bail", "Contrat de prestation", "Contrat de travail"])
    d = g.date()
    start = _shift_month(d, 1).replace(day=1)
    titles = {"Contrat de bail": {"fr": ["CONTRAT DE BAIL COMMERCIAL", "Bail professionnel", "Convention d'occupation de locaux"],
                                  "en": ["COMMERCIAL LEASE AGREEMENT", "Office lease", "Tenancy agreement"],
                                  "de": ["GEWERBEMIETVERTRAG", "Mietvertrag für Büroräume", "Pachtvertrag"]},
              "Contrat de prestation": {"fr": ["CONTRAT DE PRESTATION DE SERVICES", "Contrat de services", "Convention de prestations"],
                                        "en": ["SERVICE AGREEMENT", "Services contract", "Master services agreement"],
                                        "de": ["DIENSTLEISTUNGSVERTRAG", "Dienstvertrag", "Rahmenvertrag über Leistungen"]},
              "Contrat de travail": {"fr": ["CONTRAT DE TRAVAIL À DURÉE INDÉTERMINÉE", "Contrat de travail", "Contrat d'emploi"],
                                     "en": ["EMPLOYMENT CONTRACT", "Contract of employment", "Employment agreement"],
                                     "de": ["ARBEITSVERTRAG", "Unbefristeter Arbeitsvertrag", "Anstellungsvertrag"]}}
    other = {"Contrat de bail": g.rng.choice(g.pool["landlords"])[0],
             "Contrat de prestation": g.rng.choice(g.pool["suppliers"])[0],
             "Contrat de travail": "le salarié / the employee / der Arbeitnehmer"}[kind]
    between = {"fr": f"Entre {company} et {other}", "en": f"Between {company} and {other}",
               "de": f"Zwischen {company} und {other}"}[lang]
    lines = [g.pick(titles[kind][lang]), between]
    traps = []
    if undated:
        lines += [{"fr": "Le présent contrat prend effet à la date de sa signature.",
                   "en": "This agreement takes effect on the date of signature.",
                   "de": "Dieser Vertrag tritt mit Unterzeichnung in Kraft."}[lang],
                  {"fr": "Fait à Luxembourg, le ..............", "en": "Signed in Luxembourg on ..............",
                   "de": "Luxemburg, den .............."}[lang]]
        traps.append("unsigned_draft")
        period = NO_DATE
    else:
        until = dt.date(start.year + 3, start.month, 1) - dt.timedelta(days=1)
        lines += [{"fr": f"Le présent contrat prend effet le {g.fmt(start, lang, 'long')} jusqu'au {g.fmt(until, lang)}.",
                   "en": f"This agreement commences on {g.fmt(start, lang, 'long')} and ends on {g.fmt(until, lang)}.",
                   "de": f"Der Vertrag beginnt am {g.fmt(start, lang, 'long')} und endet am {g.fmt(until, lang)}."}[lang],
                  CLAUSES[kind][lang],
                  {"fr": f"Fait à Luxembourg, le {g.fmt(d, lang)}, en deux exemplaires.",
                   "en": f"Signed in Luxembourg on {g.fmt(d, lang)}, in two copies.",
                   "de": f"Luxemburg, den {g.fmt(d, lang)}, in zweifacher Ausfertigung."}[lang]]
        traps += ["effective_date_next_month", "end_date", "signature_at_the_end"]
        period = _period(d)
    return lines, period, kind, "", "Juridique/Contrats", traps


CLAUSES = {
    "Contrat de bail": {"fr": "Article 2. Loyer et charges. Article 3. Résiliation.",
                        "en": "Clause 2. Rent and charges. Clause 3. Termination.",
                        "de": "§ 2 Miete und Nebenkosten. § 3 Kündigung."},
    "Contrat de prestation": {"fr": "Article 2. Honoraires. Article 3. Résiliation.",
                              "en": "Clause 2. Fees. Clause 3. Termination.",
                              "de": "§ 2 Vergütung. § 3 Kündigung."},
    "Contrat de travail": {"fr": "Article 2. Rémunération et horaire. Article 3. Période d'essai.",
                           "en": "Clause 2. Salary and working hours. Clause 3. Probation.",
                           "de": "§ 2 Gehalt und Arbeitszeit. § 3 Probezeit."},
}


def insurance(g, lang, company, _):
    ins, form = g.rng.choice(g.pool["insurers"])
    year = g.rng.randint(2023, 2025)
    issued = dt.date(year - 1, 12, g.rng.randint(10, 28)) if g.chance(0.5) else dt.date(year, 1, g.rng.randint(2, 20))
    title = {"fr": ["ATTESTATION D'ASSURANCE", "Certificat d'assurance", "Attestation de garantie"],
             "en": ["CERTIFICATE OF INSURANCE", "Insurance certificate", "Confirmation of cover"],
             "de": ["VERSICHERUNGSBESTÄTIGUNG", "Versicherungsnachweis", "Deckungsbestätigung"]}[lang]
    cover = {"fr": f"Période de garantie du 01/01/{year} au 31/12/{year}",
             "en": f"Period of cover from 01/01/{year} to 31/12/{year}",
             "de": f"Versicherungszeitraum 01.01.{year} bis 31.12.{year}"}[lang]
    issued_lab = {"fr": "Délivrée le", "en": "Issued on", "de": "Ausgestellt am"}[lang]
    lines = [g.header(ins, form), g.address(), g.pick(title),
             {"fr": "Responsabilité civile professionnelle", "en": "Professional liability",
              "de": "Berufshaftpflicht"}[lang],
             f"{'Assuré' if lang == 'fr' else 'Insured' if lang == 'en' else 'Versicherungsnehmer'} : {company}",
             f"Police / Policy {g.rng.randint(100000, 999999)}", cover, f"{issued_lab} {g.fmt(issued, lang)}"]
    return lines, _period(issued), "Attestation d'assurance", ins, "Assurances", \
        ["coverage_range", "issued_previous_year"] if issued.year < year else ["coverage_range"]


def letter(g, lang, company, undated):
    kind = g.rng.choice(["tax", "vat", "social", "bank", "landlord"])
    d = g.date()
    sender, base = {"tax": (TAX_OFFICE, "Fiscal/Impôts directs"), "vat": (VAT_OFFICE, "TVA"),
                    "social": (SOCIAL_OFFICE, "Cotisations sociales"),
                    "bank": (g.rng.choice(g.pool["banks"])[0], "Banque"),
                    "landlord": (g.rng.choice(g.pool["landlords"])[0], "Courrier divers")}[kind]
    subject = {"fr": {"tax": "Demande de renseignements complémentaires", "vat": "Changement de régime de déclaration",
                      "social": "Mise à jour de vos données d'employeur", "bank": "Modification des conditions tarifaires",
                      "landlord": "Indexation du loyer"},
               "en": {"tax": "Request for additional information", "vat": "Change of filing frequency",
                      "social": "Update of your employer details", "bank": "Change to our fees",
                      "landlord": "Rent indexation"},
               "de": {"tax": "Anforderung ergänzender Unterlagen", "vat": "Änderung des Erklärungszeitraums",
                      "social": "Aktualisierung Ihrer Arbeitgeberdaten", "bank": "Änderung der Gebühren",
                      "landlord": "Indexierung der Miete"}}[lang][kind]
    re_ = {"fr": "Concerne", "en": "Subject", "de": "Betreff"}[lang]
    where = {"fr": "Luxembourg, le", "en": "Luxembourg,", "de": "Luxemburg, den"}[lang]
    answer = {"fr": "Nous vous prions de bien vouloir répondre dans un délai de 30 jours.",
              "en": "Please reply within 30 days.", "de": "Bitte antworten Sie innerhalb von 30 Tagen."}[lang]
    lines = [sender.upper() if g.chance(0.5) else sender, g.address(), company]
    if not undated:
        lines.append(f"{where} {g.fmt(d, lang)}")
    lines += [f"{re_} : {subject}", answer]
    traps = ["folder_depends_on_sender"]
    if undated:
        traps.append("letter_without_date")
    period = NO_DATE if undated else _period(d)
    return lines, period, "Courrier", sender, base, traps


BUILDERS = {"bank_statement": bank_statement, "supplier_invoice": supplier_invoice,
            "credit_note": credit_note, "sales_invoice": sales_invoice, "receipt": receipt,
            "payslip": payslip, "social": social, "vat": vat, "tax_assessment": tax_assessment,
            "articles": articles, "minutes": minutes, "contract": contract,
            "insurance": insurance, "letter": letter}

# Which kinds can come without any date, and how many of them do.
UNDATED = {"supplier_invoice": 2, "sales_invoice": 1, "receipt": 3, "contract": 3, "letter": 1}

# ------------------------------------------------------------- the noise --
LOGO_KINDS = {"supplier_invoice", "credit_note", "bank_statement", "insurance"}
OCR_SWAPS = [("e", "c"), ("l", "1"), ("o", "0"), ("i", "l"), ("m", "rn"), ("é", "e"), ("u", "n")]
_MONTH_WORDS = {m.lower() for ms in MONTHS.values() for m in ms}


def ocr_noise(g, lines):
    """Damage letters the way a phone photo does - titles included. Never the
    first two lines (the emitter), never a word holding a digit, never a month
    name: the expected answer must stay readable by a careful human."""
    out = []
    for i, line in enumerate(lines):
        if i < 2:
            out.append(line)
            continue
        words = []
        for w in line.split(" "):
            if (len(w) >= 5 and not re.search(r"\d", w) and w.lower().strip(".,:") not in _MONTH_WORDS
                    and g.chance(0.3)):
                a, b = g.rng.choice(OCR_SWAPS)
                if a in w.lower():
                    i_ = w.lower().index(a)
                    w = w[:i_] + (b.upper() if w[i_].isupper() else b) + w[i_ + 1:]
            words.append(w)
        out.append(" ".join(words))
    return out


def logo_lost(g, lines, emitter):
    """The letterhead was an image: the text layer starts with a web address,
    and the emitter's full name only survives in the footer."""
    full = lines[0]
    slug = re.sub(r"[^a-z]+", "-", emitter.lower().replace("é", "e").replace("ü", "u").replace("ö", "o")).strip("-")
    first = g.rng.choice([f"www.{slug}.lu", "[logo]", f"info@{slug}.lu"])
    footer = f"{full} - RCS B{g.rng.randint(10000, 99999)} - TVA {g.vat_id()}"
    return [first] + lines[1:] + ["", footer]


def messy_name(g, kind, lang, period_date):
    """The name the file arrived with. Some carry a date that is NOT the
    document's (the day the photo was taken or the file downloaded)."""
    n = g.rng.randint(1, 9999)
    shot = period_date + dt.timedelta(days=g.rng.randint(20, 200)) if period_date else g.date(2025, 2025)
    hint = {"supplier_invoice": {"fr": "facture", "en": "invoice", "de": "Rechnung"},
            "bank_statement": {"fr": "releve", "en": "statement", "de": "Kontoauszug"},
            "payslip": {"fr": "fiche_paie", "en": "payslip", "de": "Lohn"}}.get(kind, {}).get(lang)
    options = [(f"scan_{n:04d}.pdf", []), (f"Document ({n % 12 + 1}).pdf", []),
               (f"IMG_{shot.strftime('%Y%m%d')}_{g.rng.randint(100000, 235959)}.jpg", ["filename_date_is_photo_date"]),
               (f"WhatsApp Image {shot.isoformat()} at {g.rng.randint(8, 22)}.{g.rng.randint(10, 59)}.{g.rng.randint(10, 59)}.jpeg",
                ["filename_date_is_photo_date"]),
               (f"{shot.strftime('%Y%m%d')}_download.pdf", ["filename_date_is_download_date"]),
               (f"Copie de copie de scan{n % 7}.pdf", [])]
    if hint:
        options.append((f"{hint}_{n}.pdf", []))
    return g.rng.choice(options)


def build(split, seed, pool):
    g = Gen(split, seed, pool)
    plan = []
    for kind, count in MIX:
        dev_k = round(count * 2 / 3)
        k = dev_k if split == "dev" else count - dev_k
        undated = round(UNDATED.get(kind, 0) * k / count)
        if UNDATED.get(kind) and undated == 0 and split == "test" and kind in ("receipt", "contract"):
            undated = 1
        plan += [(kind, i < undated) for i in range(k)]
    g.rng.shuffle(plan)
    pieces = []
    for i, (kind, undated) in enumerate(plan):
        lang = g.rng.choices(["fr", "en", "de"], weights=[45, 25, 30])[0]
        company = g.rng.choice(g.pool["companies"])[0]
        lines, period, nature, emitter, base, traps = BUILDERS[kind](g, lang, company, undated)
        if emitter and kind in LOGO_KINDS and g.chance(0.25):
            lines = logo_lost(g, lines, emitter)
            traps = traps + ["logo_lost_emitter_in_footer"]
        if g.chance(0.25):
            lines = ocr_noise(g, lines)
            traps = traps + ["ocr_noise"]
        pdate = None
        if period != NO_DATE:
            pdate = dt.date(int(period[:4]), int(period[5:7]) if len(period) > 4 else 6, 15)
        fname, ftraps = messy_name(g, kind, lang, pdate)
        pieces.append({
            "id": f"{split}-{i + 1:03d}", "split": split, "kind": kind, "lang": lang,
            "company": company, "filename": fname, "text": "\n".join(lines),
            "expected": {"period": period, "nature": nature, "emitter": emitter,
                         "folder": place(base, period)},
            "traps": traps + ftraps,
        })
    return pieces


def build_all():
    return {"dev": build("dev", 11, "A"), "test": build("test", 29, "B")}


def load(split):
    path = os.path.join(DATA, f"{split}.jsonl")
    if not os.path.exists(path):
        write()
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write():
    os.makedirs(DATA, exist_ok=True)
    for split, pieces in build_all().items():
        with open(os.path.join(DATA, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for p in pieces:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"{split}: {len(pieces)} pieces")


assert set(NATURE_FOLDER)  # the schema is the single source of the natures

if __name__ == "__main__":
    write()
