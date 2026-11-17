import pandas as pd
import pytest

from readmit import ingest, synth


def test_load_tables_round_trip(tmp_path, synthetic_tables):
    synth.write_tables(synthetic_tables, tmp_path)
    tables = ingest.load_tables(tmp_path)
    assert set(tables) == {"patients", "admissions", "discharges", "labs"}
    assert pd.api.types.is_datetime64_any_dtype(tables["labs"]["charted_time"])
    assert len(tables["admissions"]) == len(synthetic_tables["admissions"])


def test_creatinine_converted_to_mg_dl(tmp_path, synthetic_tables):
    labs = synthetic_tables["labs"]
    assert (labs["unit"] == "umol/L").any()
    synth.write_tables(synthetic_tables, tmp_path)
    tables = ingest.load_tables(tmp_path)
    creat = tables["labs"].loc[tables["labs"]["lab_name"] == "creatinine"]
    assert (creat["unit"] == "mg/dL").all()
    assert creat["value"].max() < 20


def test_unknown_unit_rejected(small_tables):
    labs = small_tables["labs"].copy()
    labs.loc[0, "unit"] = "mmol/L"
    with pytest.raises(ingest.IngestError, match="unsupported units"):
        ingest.normalise_lab_units(labs)


def test_missing_column_rejected(tmp_path, synthetic_tables):
    synth.write_tables(synthetic_tables, tmp_path)
    frame = pd.read_csv(tmp_path / "discharges.csv").drop(columns=["discharge_time"])
    frame.to_csv(tmp_path / "discharges.csv", index=False)
    with pytest.raises(ingest.IngestError, match="missing columns"):
        ingest.load_tables(tmp_path)


def test_bad_timestamp_rejected(tmp_path, synthetic_tables):
    synth.write_tables(synthetic_tables, tmp_path)
    frame = pd.read_csv(tmp_path / "admissions.csv")
    frame.loc[0, "admit_time"] = "31/02/2022"
    frame.to_csv(tmp_path / "admissions.csv", index=False)
    with pytest.raises(ingest.IngestError, match="unparseable"):
        ingest.load_tables(tmp_path)


def test_discharge_before_admit_rejected(small_tables):
    small_tables["discharges"].loc[0, "discharge_time"] = pd.Timestamp("2021-12-31 00:00")
    with pytest.raises(ingest.IngestError, match="precedes"):
        ingest.validate_tables(small_tables)
