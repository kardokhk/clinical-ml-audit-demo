# readmit: 30-day readmission risk

`readmit` predicts, at the moment of hospital discharge, the probability that the patient has an unplanned (emergency) admission within the following 30 days. It ingests an EHR extract (patients, admissions, discharges, timestamped labs), builds one feature row per index admission, trains a scikit-learn pipeline, evaluates it on held-out patients and scores new admissions in batch.

The platform ships with a seeded generator for a synthetic extract. All patients are fictional and identified only by opaque ids.

## Requirements

Python 3.11 and the packages in `requirements.txt`:

```
python -m pip install -r requirements.txt
```

All commands below are run from this folder (the one containing `config.yaml`). Every command accepts `--config PATH` (default `config.yaml`) and `--log-level` before the sub-command name.

## Commands

1. Generate the synthetic data:

   ```
   python -m readmit generate
   ```

   Writes `data/raw/patients.csv`, `data/raw/admissions.csv`, `data/raw/discharges.csv`, `data/raw/labs.csv` (2,500 patients, seed from `config.yaml`) and `data/new_admissions.csv` (one admission per patient for 300 further fictional patients, already in the inference input schema).

2. Run the full train and evaluate pipeline:

   ```
   python -m readmit train
   ```

   Loads and validates `data/raw/`, builds the cohort and features, assigns patients to train or test, fits the model on the training patients, predicts every test admission, writes the predictions file and computes the metrics from that file. The metrics are also printed as JSON.

3. Score new admissions:

   ```
   python -m readmit predict --input data/new_admissions.csv --output outputs/inference_predictions.csv
   ```

   Uses the model saved by `train`. Predictions go to the path given in `--output`.

4. Run the tests:

   ```
   python -m pytest -q tests
   ```

## Output files

All written to `outputs/` (`paths.output_dir` in `config.yaml`) by `train`, except the inference output.

| File | Content |
|---|---|
| `outputs/features.csv` | One row per index admission: `patient_id`, `admission_id`, `prediction_time`, the 12 feature columns, `readmit_30d` |
| `outputs/split_assignment.csv` | `patient_id`, `admission_id`, `split` (`train` or `test`), one row per index admission |
| `outputs/predictions_test.csv` | Test-set predictions: `patient_id`, `admission_id`, `prediction_time`, `readmit_30d`, `y_prob`, `y_pred` |
| `outputs/metrics.json` | `auroc`, `auprc`, `brier`, `threshold`, `sensitivity`, `specificity`, `ppv`, `n_test`, `n_test_patients`, `prevalence_test` |
| `outputs/model.joblib` | Fitted scikit-learn pipeline (preprocessing and classifier) |
| `outputs/model_card.json` | Model type, `decision_threshold`, feature columns, training size and prevalence, seed, scikit-learn version |
| `--output` of `predict` | `patient_id`, `admission_id`, `y_prob`, `y_pred`, `decision_threshold` |

`y_prob` is the predicted probability of `readmit_30d = 1`. `y_pred` is 1 when `y_prob` is at or above the decision threshold (`model.decision_threshold` in `config.yaml`, 0.35), both in the test predictions and in inference. The metrics in `metrics.json` that depend on a threshold use the same value.

## Data model

Input tables in `data/raw/` (timestamps as `YYYY-MM-DD HH:MM:SS`, local hospital time):

| Table | Columns | Timestamp columns |
|---|---|---|
| `patients.csv` | `patient_id`, `birth_year`, `sex` (`F`, `M`) | none |
| `admissions.csv` | `admission_id`, `patient_id`, `admit_time`, `admission_type` (`emergency`, `elective`), `ward` | `admit_time` |
| `discharges.csv` | `admission_id`, `discharge_time`, `discharge_disposition` (`home`, `home_with_services`, `facility`, `died`) | `discharge_time` |
| `labs.csv` | `lab_id`, `admission_id`, `lab_name`, `value`, `unit`, `charted_time` | `charted_time` |

Lab units accepted: creatinine `mg/dL` or `umol/L`, hemoglobin `g/dL` or `g/L`, sodium `mmol/L`, wbc `10^9/L`, temperature `C`. Values are converted to the first unit listed on ingest. Any other unit is rejected.

Key columns:

- Patient id column: `patient_id`
- Admission id column: `admission_id`
- Target column: `readmit_30d`
- Prediction-time column: `prediction_time` (equal to the admission's `discharge_time`)

Cohort and label:

- Index admissions are all admissions where the patient was discharged alive.
- `readmit_30d = 1` when the same patient has an `emergency` admission whose `admit_time` falls after the index `discharge_time` and within 30 days of it (`cohort.horizon_days`). Elective admissions are not events.
- Index admissions whose 30-day follow-up window extends past the last timestamp in the extract are excluded.
- Only information available at the prediction time is used for features: labs charted between `admit_time` and `discharge_time` of the index admission, and earlier admissions of the same patient that ended before the index admission began.
- Patients, not admissions, are assigned to train or test (`split.test_fraction`, 0.25), so all admissions of one patient are in the same split.

## Inference input schema

`predict` takes a CSV with one row per admission and these columns (extra columns are ignored):

| Column | Type and unit | Allowed values | Missing allowed |
|---|---|---|---|
| `patient_id` | string | any non-empty | no |
| `admission_id` | string, unique in the file | any non-empty | no |
| `age_years` | integer, years at admission | 18 to 110 | no |
| `length_of_stay_days` | float, days from admission to discharge | 0 to 365 | no |
| `prior_admissions_365d` | integer, admissions of the same patient that ended before this admission began, within the previous 365 days | 0 to 60 | no |
| `creatinine_last` | float, mg/dL, last value during the stay | 0.1 to 20 | yes |
| `hemoglobin_last` | float, g/dL, last value during the stay | 3 to 25 | yes |
| `sodium_last` | float, mmol/L, last value during the stay | 100 to 180 | yes |
| `wbc_last` | float, 10^9/L, last value during the stay | 0.1 to 200 | yes |
| `temperature_max_c` | float, degrees Celsius, maximum during the stay | 30 to 45 | yes |
| `sex` | string | `F`, `M` | no |
| `admission_type` | string | `emergency`, `elective` | no |
| `ward` | string | `cardiology`, `general_medicine`, `orthopaedics`, `surgery` | no |
| `discharge_disposition` | string | `home`, `home_with_services`, `facility` | no |

Missing lab values are imputed by the saved pipeline with the training-set median (plus a missingness indicator), exactly as in training. Rows that break the schema (missing column, value out of range, unknown category, duplicate `admission_id`) stop the run with an error that names the problem; nothing is written in that case.

## Configuration

`config.yaml` holds the seed, paths, synthetic data size and date range, the 30-day horizon, the 365-day lookback for `prior_admissions_365d`, the test fraction and the model settings (random forest, 200 trees, maximum depth 8, minimum leaf size 20, decision threshold 0.35). A single seed (`seed`) drives the data generator, the patient split and the model, so repeated runs on the same inputs give identical outputs.

## Package layout

```
readmit/
  schema.py     column names, feature list, ranges, category levels
  config.py     config loading and validation
  synth.py      synthetic EHR generator
  ingest.py     table loading, validation, unit normalisation
  cohort.py     index admissions, readmission label, follow-up exclusion
  features.py   feature construction and the feature-frame validator
  split.py      patient-level train/test assignment
  model.py      scikit-learn pipeline, fit, save, load
  evaluate.py   metrics from the predictions file
  inference.py  batch scoring service
  pipeline.py   orchestration of generate, train and predict
  cli.py        command-line interface
tests/          pytest suite
```
