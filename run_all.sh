#!/usr/bin/env bash
# Audit all four stand-in platforms and score them against their defect lists.
# Usage: PY=/path/to/python VULN_PY="/path/to/python-with-pip-audit" ./run_all.sh results/<run-name> [mutants]
# PY must have the packages in requirements.txt. VULN_PY is optional (network scan with pip-audit).
set -euo pipefail
cd "$(dirname "$0")"
OUT=${1:?results directory}
MUT=${2:-40}
PY=${PY:-python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT"
for t in platform_a platform_a_clean platform_b platform_b_clean; do
  args=(targets/$t.yaml --out "$OUT/$t" --python "$PY" --mutants "$MUT")
  [ -n "${VULN_PY:-}" ] && args+=(--vuln-python "$VULN_PY")
  "$PY" -m mlaudit.audit "${args[@]}"
done
for p in a b; do
  m=stand_in/manifests/platform_${p}_defects.json
  if [ -f "$m" ]; then
    "$PY" -m mlaudit.score --manifest "$m" --defective "$OUT/platform_$p" --clean "$OUT/platform_${p}_clean" --out "$OUT/score_$p.json"
  else
    echo "no manifest for platform $p yet (sealed until after the frozen run)"
  fi
done
