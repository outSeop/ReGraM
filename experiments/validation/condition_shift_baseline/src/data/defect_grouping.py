"""Canonical LOCO defect grouping helpers.

The grouping is for post-hoc analysis only. Scoring code should not use
defect groups or GT area to choose favorable samples or compute anomaly scores.
"""

from __future__ import annotations

from typing import Any, Iterable


def defect_group_for_name(defect_name: str) -> str:
    """Map a LOCO defect name to a stable coarse analysis group."""
    name = str(defect_name).lower()
    if name.startswith("missing") or "underflow" in name:
        return "missing_or_under_count"
    if "overflow" in name or "extra" in name or "count" in name:
        return "extra_or_over_count"
    if "swapped" in name or "misplaced" in name:
        return "relation_or_layout"
    if "wrong_ratio" in name or "mixed" in name or "ratio" in name:
        return "composition_or_ratio"
    if "damaged" in name or "crushed" in name or "contamination" in name:
        return "local_damage_or_contamination"
    return "other_logical"


def defect_group_for_names(defect_names: Iterable[str]) -> str:
    """Return a deterministic comma-joined group label for multiple defects."""
    groups = sorted({defect_group_for_name(name) for name in defect_names})
    return ",".join(groups) if groups else "other_logical"


def defect_config_rows(config: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert a defects_config.json list to canonical table rows."""
    rows = []
    for item in config:
        defect_name = str(item.get("defect_name", item.get("name", "unknown")))
        rows.append(
            {
                "defect_name": defect_name,
                "pixel_value": int(item.get("pixel_value", -1)),
                "rough_defect_group": defect_group_for_name(defect_name),
            }
        )
    return rows
