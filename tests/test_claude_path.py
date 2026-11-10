"""The Claude path end to end with a fake client: no key, no network, no cost."""
import json
import sys
import types
from types import SimpleNamespace

from filing_bench import bench, corpus, llm

FORBIDDEN = {"temperature", "top_p", "top_k"}
ANSWER = {"period": "2025-03", "nature": "Facture", "emitter": "Papeterie Weber", "folder": "Achats"}


class FakeMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="thinking", thinking=""),
                     SimpleNamespace(type="text", text=json.dumps(ANSWER))],
            usage=SimpleNamespace(input_tokens=900, output_tokens=60))


def fake_sdk(monkeypatch, tmp_path):
    fake = SimpleNamespace(messages=FakeMessages())
    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=lambda: fake))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-tests")
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)
    monkeypatch.setattr(llm, "CACHE", str(tmp_path / "cache"))
    return fake


def check_request(kw):
    assert not FORBIDDEN & kw.keys(), "current models reject sampling parameters"
    assert kw["messages"][-1]["role"] == "user", "no assistant prefill"
    assert kw["model"] == "claude-haiku-5-5"


def test_default_model_is_haiku(monkeypatch, tmp_path):
    fake_sdk(monkeypatch, tmp_path)
    assert llm.ClaudeModel().model == "claude-haiku-5-5"
    monkeypatch.setenv("CLAUDE_MODEL", "claude-sonnet-5-5")
    assert llm.ClaudeModel().model == "claude-sonnet-5-5"


def test_estimate_is_offline_and_small(monkeypatch, tmp_path):
    fake = fake_sdk(monkeypatch, tmp_path)
    pieces = corpus.load("dev") + corpus.load("test")
    e = llm.ClaudeModel().estimate(pieces)
    assert e["calls"] == len(pieces) == 150 and not fake.messages.calls
    assert 0 < e["expected_usd"] < e["worst_usd"] < 1.0


def test_one_command_adds_the_claude_row(monkeypatch, tmp_path, capsys):
    """bench --claude --yes (local models left out): estimate printed, Claude row in results.md."""
    fake = fake_sdk(monkeypatch, tmp_path)
    monkeypatch.setattr(bench, "ROOT", str(tmp_path))
    monkeypatch.setattr(bench, "RESULTS", str(tmp_path / "results"))

    assert bench.main(["--claude", "--yes", "--models"]) in (None, 0)

    out = capsys.readouterr().out
    assert "Claude (claude-haiku-5-5): 150 pieces to send, 0 already cached" in out
    assert len(fake.messages.calls) == 150
    for kw in fake.messages.calls:
        check_request(kw)
    report = (tmp_path / "results.md").read_text(encoding="utf-8")
    row = next(line for line in report.splitlines() if line.startswith("| claude-haiku-5-5 |"))
    assert "per 1000" in row
    assert "Claude was **not** run" not in report

    # Second run: everything is cached, so no key, no call, and the row stays.
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    fake.messages.calls.clear()
    bench.main(["--models"])
    assert not fake.messages.calls
    assert "| claude-haiku-5-5 |" in (tmp_path / "results.md").read_text(encoding="utf-8")


def test_no_confirmation_sends_nothing(monkeypatch, tmp_path, capsys):
    fake = fake_sdk(monkeypatch, tmp_path)
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert bench.main(["--claude", "--models"]) == 1
    assert not fake.messages.calls and "nothing was sent" in capsys.readouterr().out
