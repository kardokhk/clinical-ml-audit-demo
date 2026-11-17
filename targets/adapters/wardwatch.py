"""Build a WardWatch inference input (four CSV tables) for a list of admissions, from the README format."""
from pathlib import Path

import pandas as pd


def make_input(sandbox: Path, ids: list, out_dir: Path) -> Path:
    ids = set(map(str, ids))
    adm = pd.read_csv(sandbox / "data" / "admissions.csv", dtype=str)
    adm = adm[adm["admission_id"].isin(ids)].drop(columns=["discharge_time", "icu_transfer_time", "deteriorated_48h"])
    pat = pd.read_csv(sandbox / "data" / "patients.csv", dtype=str)
    pat = pat[pat["patient_id"].isin(set(adm["patient_id"]))]
    out_dir.mkdir(parents=True, exist_ok=True)
    adm.to_csv(out_dir / "admissions.csv", index=False)
    pat.to_csv(out_dir / "patients.csv", index=False)
    for name in ("vitals", "labs"):
        t = pd.read_csv(sandbox / "data" / f"{name}.csv", dtype={"admission_id": str})
        t[t["admission_id"].isin(ids)].to_csv(out_dir / f"{name}.csv", index=False)
    return out_dir
