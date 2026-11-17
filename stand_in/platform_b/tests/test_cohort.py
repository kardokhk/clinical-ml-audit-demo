import pandas as pd

from readmit import cohort


def test_label_counts_emergency_within_30_days(small_tables):
    stays = cohort.build_stays(small_tables)
    index = cohort.select_index_admissions(stays)
    labelled = cohort.label_readmissions(index, stays, 30).set_index("admission_id")
    assert labelled.loc["A1", "readmit_30d"] == 1   # A2 starts 15 days after A1 discharge
    assert labelled.loc["A2", "readmit_30d"] == 0   # A3 is elective
    assert labelled.loc["A3", "readmit_30d"] == 0
    assert labelled.loc["A4", "readmit_30d"] == 0


def test_elective_readmission_is_not_an_event(small_tables):
    small_tables["admissions"].loc[2, "admit_time"] = pd.Timestamp("2022-02-01 10:00")
    small_tables["discharges"].loc[2, "discharge_time"] = pd.Timestamp("2022-02-02 10:00")
    stays = cohort.build_stays(small_tables)
    labelled = cohort.label_readmissions(cohort.select_index_admissions(stays), stays, 30)
    assert labelled.set_index("admission_id").loc["A2", "readmit_30d"] == 0


def test_in_hospital_deaths_excluded(small_tables):
    small_tables["discharges"].loc[3, "discharge_disposition"] = "died"
    index = cohort.select_index_admissions(cohort.build_stays(small_tables))
    assert "A4" not in set(index["admission_id"])


def test_incomplete_followup_excluded(small_tables):
    stays = cohort.build_stays(small_tables)
    labelled = cohort.label_readmissions(cohort.select_index_admissions(stays), stays, 30)
    kept = cohort.exclude_incomplete_followup(labelled, pd.Timestamp("2022-04-10"), 30)
    assert "A3" not in set(kept["admission_id"])
    assert {"A1", "A2", "A4"} <= set(kept["admission_id"])
