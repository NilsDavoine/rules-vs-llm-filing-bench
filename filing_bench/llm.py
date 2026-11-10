"""Language-model proposers: a small open model run locally, and Claude (optional).

Both get the same prompt: the convention, the closed lists of natures and
folders, two invented examples, then the piece. Both answer one JSON object.
The answer goes through the same `compose_name`, `place` and guard as the
rules, so the comparison is about reading the document, nothing else.

Answers are cached in results/cache/<model>.jsonl, keyed by a hash of the
prompt: re-running the report does not re-run the models, and the cache is
the raw evidence of what each model said.
"""
import hashlib
import json
import os
import re
import time

from .schema import NATURES, NO_DATE, PERMANENT, YEARLY, Proposal, fold, place, valid_period

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "results", "cache")

LOCAL_MODELS = {
    # name: (Hugging Face repo, pinned revision)
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
}

BASE_FOLDERS = YEARLY + PERMANENT

SYSTEM = f"""You file accounting documents for a fiduciary in Luxembourg.
For the document given, answer ONE JSON object with four keys:
"period": the period the document covers, written YYYY-MM. Invoice, credit note, receipt, letter: its date of issue (not the due date). Bank statement: the end of the statement period. Payslip, social contributions: the month paid for. VAT return: the last month of its quarter or month; an annual return gives YYYY. Tax assessment: the tax year, YYYY. Contract: the signature date. Minutes, articles: the date of the meeting or deed. Insurance certificate: its issue date. If the document shows no date, write "{NO_DATE}". Never guess a date.
"nature": exactly one of {json.dumps(list(NATURES), ensure_ascii=False)}. "Facture client" is an invoice issued BY the client company named below; "Facture" is an invoice it received.
"emitter": who issued the document (company or administration name without legal form such as S.A., S.à r.l., GmbH, Ltd); for "Facture client", the customer billed. Empty string for payslips, VAT returns, tax assessments, contracts, minutes, articles and receipts.
"folder": exactly one of {json.dumps(BASE_FOLDERS, ensure_ascii=False)}. A letter goes to the folder of its sender (tax office: "Fiscal/Impôts directs", VAT office: "TVA", social office: "Cotisations sociales", bank: "Banque"), otherwise "Courrier divers".
Answer with the JSON object only."""

EXAMPLES = [
    ("Client company: Exemple Conseil\nFile name: scan_0001.pdf\n---\nGARAGE MARTIN S.A.\nFACTURE 22-0042\nClient : Exemple Conseil\n"
     "Date : 03/02/2022\nÉchéance : 05/03/2022\nTotal TTC : 351,00 EUR",
     {"period": "2022-02", "nature": "Facture", "emitter": "Garage Martin", "folder": "Achats"}),
    ("Client company: Exemple Conseil\nFile name: IMG_20220930_101010.jpg\n---\nEXEMPLE CONSEIL S.A.\nLohnabrechnung\nMonat: Juli 2022\n"
     "Personalnr. 0007\nGeburtsdatum 01.01.1990\nAuszahlung 2 100,00",
     {"period": "2022-07", "nature": "Fiche de salaire", "emitter": "", "folder": "Salaires"}),
]

MAX_TEXT = 1800   # characters of the document given to a model


def user_message(piece):
    return (f"Client company: {piece['company']}\nFile name: {piece['filename']}\n---\n"
            f"{piece['text'][:MAX_TEXT]}")


def messages(piece):
    msgs = [{"role": "system", "content": SYSTEM}]
    for u, a in EXAMPLES:
        msgs += [{"role": "user", "content": u},
                 {"role": "assistant", "content": json.dumps(a, ensure_ascii=False)}]
    msgs.append({"role": "user", "content": user_message(piece)})
    return msgs


def parse(raw):
    """Model text -> Proposal. Anything unusable becomes an empty field, never a guess."""
    p = Proposal(raw=raw)
    m = re.search(r"\{.*?\}", raw or "", re.S)
    if not m:
        return p
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return p
    if not isinstance(data, dict):
        return p
    period = str(data.get("period", "")).strip()
    p.period = period if valid_period(period) else ""
    wanted = fold(str(data.get("nature", "")))
    p.nature = next((n for n in NATURES if fold(n) == wanted), "")
    p.emitter = str(data.get("emitter", "") or "").strip()
    base = fold(str(data.get("folder", "")))
    base = next((f for f in BASE_FOLDERS if fold(f) == base), "")
    if not p.period:
        p.period = NO_DATE
    p.folder = place(base, p.period) if base else ""
    return p


class _Cache:
    def __init__(self, model):
        os.makedirs(CACHE, exist_ok=True)
        self.path = os.path.join(CACHE, f"{model}.jsonl")
        self.rows = {}
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    self.rows[row["key"]] = row

    @staticmethod
    def key(msgs):
        return hashlib.sha256(json.dumps(msgs, ensure_ascii=False).encode()).hexdigest()[:20]

    def get(self, msgs):
        return self.rows.get(self.key(msgs))

    def put(self, msgs, row):
        row["key"] = self.key(msgs)
        self.rows[row["key"]] = row
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class LocalModel:
    """A small open instruct model through transformers, greedy decoding."""

    def __init__(self, name):
        self.name = name
        self.repo, self.revision = LOCAL_MODELS[name]
        self.cache = _Cache(name)
        self._model = None
        self.device = None

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._tok = AutoTokenizer.from_pretrained(self.repo, revision=self.revision)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.repo, revision=self.revision, torch_dtype=dtype).to(self.device).eval()
        self._torch = torch

    def unload(self):
        """Free the GPU before the next model (4 GB do not hold two)."""
        if self._model is not None:
            del self._model
            self._model = None
            self._torch.cuda.empty_cache()

    def propose(self, piece):
        msgs = messages(piece)
        row = self.cache.get(msgs)
        if row is None:
            if self._model is None:
                self._load()
            prompt = self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = self._tok(prompt, return_tensors="pt").to(self.device)
            t0 = time.perf_counter()
            with self._torch.no_grad():
                out = self._model.generate(**ids, max_new_tokens=96, do_sample=False,
                                           pad_token_id=self._tok.eos_token_id)
            seconds = time.perf_counter() - t0
            new = out[0][ids["input_ids"].shape[1]:]
            row = {"id": piece["id"], "raw": self._tok.decode(new, skip_special_tokens=True),
                   "seconds": seconds, "tokens_in": int(ids["input_ids"].shape[1]),
                   "tokens_out": int(len(new)), "device": self.device}
            self.cache.put(msgs, row)
        p = parse(row["raw"])
        p.confidence = 1.0
        return p, row


# USD per million tokens, from Anthropic's public price list (October 2026), prompts up to 100K tokens.
CLAUDE_PRICES = {"claude-opus-5-5": (4.00, 20.00), "claude-sonnet-5-5": (2.00, 10.00),
                 "claude-haiku-5-5": (0.10, 0.50)}
CLAUDE_DEFAULT = "claude-haiku-5-5"   # the cheapest current model; CLAUDE_MODEL picks another
CLAUDE_MAX_TOKENS = 4000
CHARS_PER_TOKEN = 3.0       # deliberately low (FR/DE text, JSON): over-estimates the input
EXPECTED_OUT_TOKENS = 300   # one short JSON object plus a little thinking at effort "low"


class ClaudeModel:
    """Optional. Runs only with --claude and ANTHROPIC_API_KEY; never needed to reproduce the bench.

    No temperature/top_p/top_k and no assistant prefill: current models reject both
    with a 400. The few-shot examples end on a user turn, so nothing is prefilled."""

    def __init__(self, model=None, client=None):
        self.model = model or os.environ.get("CLAUDE_MODEL", CLAUDE_DEFAULT)
        self.name = self.model
        self.cache = _Cache(self.model)
        self._client = client

    @staticmethod
    def available():
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def missing(self, pieces):
        """Pieces with no cached answer: the only ones a run would pay for."""
        return [p for p in pieces if self.cache.get(messages(p)) is None]

    def estimate(self, pieces):
        """Offline cost estimate for the pieces not cached yet: no network, no key."""
        price_in, price_out = CLAUDE_PRICES.get(self.model, (0.0, 0.0))
        todo = self.missing(pieces)
        tok_in = sum(sum(len(m["content"]) for m in messages(p)) / CHARS_PER_TOKEN for p in todo)
        exp_out, worst_out = EXPECTED_OUT_TOKENS * len(todo), CLAUDE_MAX_TOKENS * len(todo)
        usd = lambda i, o: (i * price_in + o * price_out) / 1e6
        return {"calls": len(todo), "cached": len(pieces) - len(todo), "tokens_in": tok_in,
                "expected_out": exp_out, "expected_usd": usd(tok_in, exp_out),
                "worst_out": worst_out, "worst_usd": usd(tok_in, worst_out),
                "price_in": price_in, "price_out": price_out}

    def propose(self, piece):
        msgs = messages(piece)
        row = self.cache.get(msgs)
        if row is None:
            if self._client is None:
                import anthropic
                self._client = anthropic.Anthropic()
            t0 = time.perf_counter()
            resp = self._client.messages.create(
                model=self.model, max_tokens=CLAUDE_MAX_TOKENS, system=msgs[0]["content"],
                messages=msgs[1:], output_config={"effort": "low"})
            seconds = time.perf_counter() - t0
            text = "" if resp.stop_reason == "refusal" else "".join(
                b.text for b in resp.content if b.type == "text")
            row = {"id": piece["id"], "raw": text, "seconds": seconds,
                   "tokens_in": resp.usage.input_tokens, "tokens_out": resp.usage.output_tokens,
                   "device": "api", "stop_reason": resp.stop_reason}
            self.cache.put(msgs, row)
        return parse(row["raw"]), row

    def cost(self, row):
        price_in, price_out = CLAUDE_PRICES.get(self.model, (0.0, 0.0))
        return (row["tokens_in"] * price_in + row["tokens_out"] * price_out) / 1e6
