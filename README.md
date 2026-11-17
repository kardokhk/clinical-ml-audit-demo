# Clinical ML audit demo

These are read-only audit checks for Python clinical machine learning repositories. They were tested on stand-in platforms with planted defects, and they cover data leakage, model execution integrity, failure handling, reproducibility, dependencies, automated tests and integration between components.

Kardokh Kakabra built this in November 2026 as a demonstration. All data are synthetic: each stand-in platform generates its own fictional patients from a fixed seed.

## Results

I developed the checks on platform A, a ward deterioration model. Platform B is a different code base with a different task (30-day readmission). It was written separately, and its list of 16 planted defects stayed sealed until after one frozen run (tag `frozen-test`, commit `2f8a784`). I wrote platform B's audit settings from its README alone.

| Platform B, held out | Frozen run | After post-hoc fixes |
|---|--:|--:|
| Planted defects found (hand-checked) | 12 of 16 | 14 of 16 |
| Critical or High defects found | 7 of 9 | 8 of 9 |
| Found by ruff and bandit alone | 2 of 16 | 2 of 16 |
| Critical or High findings on the defect-free twin | 7 | 0 |
| Medium or Low findings on the defect-free twin | 20 | 15 |

The frozen run is the held-out result. On it, the 95% Wilson interval for 12 of 16 runs from 51% to 90%. That interval is wide because 16 defects is a small sample.

The frozen run had two weaknesses:

- **Missed defects.** It missed a count feature that included admissions after the prediction time (Critical), a served alert threshold read from the wrong key (High), a test assertion that is always true, and a cohort step that only the tests call.
- **False alarms.** It raised 7 High false alarms on the defect-free twin. All of them came from the config-key check, which did not follow a config section passed into a function under another name, or a key the code sets itself.

The post-hoc fixes are in a separate commit and address the false alarms, the threshold and the uncalled step. Because they were made after the defect list was opened, the right-hand column is not a held-out result. Two defects are still missed: the future-admissions count and the always-true assertion.

On platform A, the development platform, 15 of 15 planted defects were found, with 0 Critical or High findings on its defect-free twin.

Each audit hashed every file of the audited folder before and after the run. No file was added, removed or changed in any of the eight audits.

Scoring rules are in `EVALUATION.md`. Per-defect results are in `results/<run>/score_*.json`, the hand checks in `results/<run>/adjudication.json`, and every quoted number in `results/summary.json`.

## What the audit does

| Area | Checks |
|---|---|
| System status | Inventory of modules, tests, entry points and config; internal import map |
| Data leakage | Patients in both train and test; preprocessing fitted before the split; splits without patient grouping; a probe that deletes every record timed after the prediction time and reruns the pipeline |
| Model execution integrity | Reported metrics recomputed from saved predictions; alerts checked against the reported threshold; validated scores compared with scores from the inference path; feature-name warnings; unverified model loading |
| Failure handling | Wrong units, out-of-range values, text in numeric fields, missing columns, unknown categories and duplicate ids sent to the inference command; broad exception handlers; silent coercion |
| Reproducibility | Two full runs under different hash seeds, compared file by file; unseeded estimators and generators; set ordering; unsorted file listings; dependence on the system clock |
| Dependencies | Exact pins, undeclared imports, unused declarations, lock files, known vulnerabilities (pip-audit) |
| Tests | Test run with coverage; tests without assertions; skipped tests; modules without tests; functions only the tests call; a seeded sample of 40 mutants |
| Integration | Documented commands that fail; config keys read but not defined; code never called |

All execution happens in disposable copies. The audited repository is never edited.

## Run it

```
python -m pip install -r requirements.txt
PY=python ./run_all.sh results/my-run 40
python -m mlaudit.register results/my-run/platform_b findings.xlsx
```

For the optional network scan of pinned dependencies, set `VULN_PY` to a Python with pip-audit.

To audit another repository, write two short files from that repository's documentation:

- a target file like `targets/platform_b.yaml`, giving the commands, output paths, and id and time columns;
- for the inference checks, an adapter like `targets/adapters/readmit.py`.

Known-answer tests for the checks: `python -m pytest -q tests`.

## Limits

- The planted defects are a sample of known defect types, not a random sample of real-world defects.
- The person who wrote the checks also chose the defect categories. Platform B's code and its specific defects were written separately and kept sealed, but this is not a fully blind test.
- Automated checks find common faults. A real audit adds a reading of the critical code path, which is where defects like the future-admissions count are found.
- Passing these checks does not make a model clinically valid.

## Licence

MIT
