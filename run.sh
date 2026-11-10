#!/usr/bin/env bash
# One command: corpus, tests, mutation check, benchmark. Pass --no-llm to skip the models.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
$PY -m filing_bench.corpus
$PY -m pytest
$PY scripts/mutation_check.py > /dev/null && echo "mutation check: every mutant caught (results/mutation.md)"
$PY -m filing_bench.bench "$@"
