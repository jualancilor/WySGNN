#!/usr/bin/env bash
# One-command reproduction of the WySGNN ablation + sample-efficiency study.
set -euo pipefail
cd "$(dirname "$0")/.."

# 1. (optional) fetch the real JARVIS piezoelectric dataset; safe to skip.
# python scripts/fetch_data.py || true

# 2. unit tests must be green before any training (execution gates).
python -m pytest tests/ -q

# 3. run the full ablation + sample-efficiency study; writes results/*.json + metrics.csv
python scripts/run_ablation.py

# 4. regenerate paper numbers/tables/figure from committed results, then compile.
python scripts/make_paper_macros.py
python scripts/make_figures.py
if command -v pdflatex >/dev/null 2>&1; then
  (cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 \
     && bibtex main >/dev/null 2>&1 \
     && pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 \
     && pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 \
     && echo "paper/main.pdf built")
fi

echo "Done. See results/ablation.json, results/sample_efficiency.json, results/metrics.csv, paper/main.pdf"
