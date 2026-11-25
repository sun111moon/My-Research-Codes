#!/usr/bin/env bash
#
# tests/run_all.sh -- cheap self-consistency suite for the DeltScan package.
#
# Runs the four Tier A test cases of docs/TEST_CASES.md (no long integrations),
# stores their full output under tests/out/ and reports PASS/FAIL against the
# reference values of release v1.0.0.
#
# Usage:  bash tests/run_all.sh
# Exit:   0 = every automated check passed, 1 = at least one failed.
#
# Override the interpreter with, e.g.,  PYTHON=python3.12 bash tests/run_all.sh

set -u

cd "$(dirname "$0")/.." || exit 1
PY=${PYTHON:-python3}
OUT=tests/out
mkdir -p "$OUT"

fail=0
pass=0

run() {                     # run <id> <command ...>
  local id=$1; shift
  printf '=== %s: %s ===\n' "$id" "$*" | tee "$OUT/$id.log"
  "$@" >>"$OUT/$id.log" 2>&1
  local rc=$?
  cat "$OUT/$id.log"
  if [ "$rc" -ne 0 ]; then
    echo "[$id] NON-ZERO EXIT ($rc)"
    fail=$((fail + 1))
  fi
  echo
}

check() {                   # check <id> <fixed-string> <description>
  if grep -qF -- "$2" "$OUT/$1.log"; then
    printf '  PASS  %s\n' "$3"
    pass=$((pass + 1))
  else
    printf '  FAIL  %s\n' "$3"
    printf '        (pattern not found: %s)\n' "$2"
    fail=$((fail + 1))
  fi
}

echo "DeltScan self-consistency suite"
echo "interpreter: $($PY -V 2>&1)"
echo

# ---------------------------------------------------------------- TC-01
run TC-01 "$PY" 04_same_field_control.py --selftest
check TC-01 "selftest: PASS" "TC-01 resampling self-test reports PASS"
check TC-01 "4.257301"       "TC-01 reference mean intensity at N_ref = 16384"

# ---------------------------------------------------------------- TC-02
run TC-02 "$PY" 06_analyze_ensemble.py --npz ensemble_results_1000.npz
check TC-02 "23.308" "TC-02 Table 4 median kurtosis at N = 4096"
check TC-02 "2.645"  "TC-02 Table 4 median kurtosis at N = 16384"
check TC-02 "99.5%"  "TC-02 Table 4 tail fraction at N = 4096"
check TC-02 "1.2%"   "TC-02 Table 4 tail fraction at N = 16384"

# ---------------------------------------------------------------- TC-03
run TC-03 "$PY" 07_bootstrap_statistics.py
check TC-03 "23.308" "TC-03 median kurtosis at N = 4096"
check TC-03 "0.00%"  "TC-03 apparent-gap fraction vanishes at m = 1000"

# ---------------------------------------------------------------- TC-04
run TC-04 "$PY" 08_bootstrap_ci.py
check TC-04 "23.0343" "TC-04 bootstrap 95% CI, lower end at N = 4096"
check TC-04 "2.6641"  "TC-04 bootstrap 95% CI, upper end at N = 16384"

echo "-------------------------------------------------------------"
echo "checks passed: $pass    checks failed: $fail"
echo
echo "Tier B tests (01 --quick, 02, 03 --all, 05 --smoke) are not run here;"
echo "see section 4 of docs/TEST_CASES.md for their commands and criteria."
echo

if [ "$fail" -eq 0 ]; then
  echo "ALL AUTOMATED CHECKS PASSED"
  exit 0
else
  echo "SOME CHECKS FAILED -- full output in $OUT/"
  exit 1
fi
