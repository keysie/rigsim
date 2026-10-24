import json
from pathlib import Path

from app.measurements import aggregate, match, validate_record

FOLDER = Path(__file__).resolve().parent.parent.parent / "measurements"
RECORDS = [json.loads(p.read_text()) for p in FOLDER.glob("*.json") if p.name != "schema.json"]


def test_shipped_records_validate():
    assert RECORDS
    for r in RECORDS:
        v = validate_record(r)
        assert v["ok"], v["errors"]


def test_clusters_and_match():
    cl = aggregate(RECORDS)
    assert len(cl) == 3
    strata = next(r for r in RECORDS if r["engine"]["engine"] == "strata")   # glob order differs by filesystem
    m = match(cl, strata["machine"], strata["engine"])
    assert {c["key"]["streams"] for c in m} == {1, 4}


def test_physical_bound_and_missing_parts():
    bad = json.loads(json.dumps(next(r for r in RECORDS if r["engine"]["engine"] == "vllm")))
    bad["results"] = {"decode_1": 400}
    assert any("memory-bus bound" in e for e in validate_record(bad)["errors"])
    bad["engine"]["model_id"] = "nope"
    assert any("unknown model" in e for e in validate_record(bad)["errors"])
