"""Finding records and severity scale."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

SEVERITIES = ["Critical", "High", "Medium", "Low"]
CATEGORIES = [
    "leakage", "execution_integrity", "failure_handling", "reproducibility",
    "dependency", "testing", "integration",
]

SEVERITY_MEANING = {
    "Critical": "Can change patient-level outputs silently or invalidates reported performance.",
    "High": "Blocks reproduction or independent evaluation, or misreports results.",
    "Medium": "Weakens evidence or maintainability; fix before clinical evaluation.",
    "Low": "Hygiene item or point to confirm with the team.",
}


@dataclass
class Finding:
    check: str
    category: str
    severity: str
    title: str
    detail: str
    recommendation: str
    file: str | None = None
    line: int | None = None
    evidence: dict = field(default_factory=dict)
    kind: str = "static"          # static, dynamic or baseline
    related_files: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity {self.severity}")
        if self.kind != "baseline" and self.category not in CATEGORIES:
            raise ValueError(f"unknown category {self.category}")

    def to_dict(self) -> dict:
        return asdict(self)


def sort_key(f: Finding) -> tuple:
    return (SEVERITIES.index(f.severity), CATEGORIES.index(f.category) if f.category in CATEGORIES else 99,
            f.file or "", f.line or 0, f.check)


def number(findings: list[Finding], prefix: str = "F") -> list[dict]:
    out = []
    for i, f in enumerate(sorted(findings, key=sort_key), start=1):
        d = f.to_dict()
        d["id"] = f"{prefix}{i:02d}"
        out.append(d)
    return out
