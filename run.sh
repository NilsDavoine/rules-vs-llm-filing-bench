#!/usr/bin/env bash
# One command: corpus, tests, mutation check, benchmark, chart.
# --no-llm skips the models; --claude also asks Claude (cost estimate and confirmation first).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
$PY -m filing_bench.corpus
$PY -m pytest
$PY scripts/mutation_check.py > /dev/null && echo "mutation check: every mutant caught (results/mutation.md)"
$PY -m filing_bench.bench "$@"
$PY scripts/plot_results.py 2>/dev/null || echo "chart skipped (pip install -r requirements-plot.txt)"
