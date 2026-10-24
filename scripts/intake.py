#!/usr/bin/env python3
"""Issue intake: read a measurement or part submission from an issue body, validate it, and write the file
the pull request will carry. Used by .github/workflows/intake.yml; also runnable by hand:

    python3 scripts/intake.py measurement issue.md     # writes measurements/<id>.json, prints a report
    python3 scripts/intake.py part issue.md            # appends to the catalog, prints a report

Exit code 0 = accepted, 1 = rejected (the report explains why). The report goes back to the issue as a comment.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.measurements import validate_record  # noqa: E402


def json_block(body: str) -> dict:
    """The first JSON object in the issue body (inside a ```json fence or bare)."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", body, re.S) or re.search(r"(\{.*\})", body, re.S)
    if not m:
        raise ValueError("no JSON object found in the issue body")
    return json.loads(m.group(1))


def measurement(body: str) -> tuple[bool, str]:
    rec = json_block(body)
    v = validate_record(rec)
    lines = []
    if v["errors"]:
        lines.append("**Rejected.** The record did not validate:")
        lines += [f"- {e}" for e in v["errors"]]
        lines.append("\nFix the record in the app (Export measurement → Check) and edit this issue.")
        return False, "\n".join(lines)
    path = ROOT / "measurements" / f"{v['id']}.json"
    if path.exists():
        return False, f"**Already present:** `measurements/{path.name}` exists with the same key, date and results."
    path.write_text(json.dumps(rec, indent=1) + "\n")
    lines.append(f"**Accepted** as `measurements/{path.name}`.")
    if v["warnings"]:
        lines.append("Warnings (the record is accepted, the trust score is lower):")
        lines += [f"- {w}" for w in v["warnings"]]
    d = v.get("derived") or {}
    if d.get("decode_1") and rec["results"].get("decode_1"):
        err = (d["decode_1"] - rec["results"]["decode_1"]) / rec["results"]["decode_1"]
        lines.append(f"\nFor reference, the model's estimate for this setup is {d['decode_1']:.0f} tok/s writing alone "
                     f"({err:+.0%} vs your measurement) and {d['prefill']:.0f} tok/s reading.")
    return True, "\n".join(lines)


def part(body: str) -> tuple[bool, str]:
    import os
    from app.partcheck import check_cpu, check_gpu, check_source
    spec = json_block(body)
    kind = "cpu" if "mem_channels" in spec else "gpu" if "vram_gb" in spec else None
    if kind is None:
        return False, "**Rejected.** Could not tell CPU from GPU: a CPU spec needs `mem_channels`, a GPU spec `vram_gb`."
    errors, warnings = (check_cpu if kind == "cpu" else check_gpu)(spec)
    url = re.search(r"https?://\S+", body)
    if not url:
        errors.append("a spec-sheet URL is required")
    else:
        e2, w2 = check_source(url.group(0).rstrip(").,"), str(spec.get("name", "")), fetch=os.environ.get("INTAKE_FETCH") == "1")
        errors += e2
        warnings += w2
    if errors:
        return False, "**Rejected.** The part did not pass the checks:\n" + "\n".join(f"- {e}" for e in errors)
    p = ROOT / "backend" / "app" / "catalog" / "proposed.json"
    data = json.loads(p.read_text())
    spec = {k: v for k, v in spec.items()}
    spec["source"] = url.group(0)
    data[f"{kind}s"].append(spec)
    p.write_text(json.dumps(data, indent=1) + "\n")
    lines = [f"**Accepted** as a proposed {kind.upper()} `{spec['id']}` (quarantined: shown in the app as community-unverified, "
             "not used for calibration; a maintainer promotes it into the catalog)."]
    if warnings:
        lines.append("Warnings:")
        lines += [f"- {w}" for w in warnings]
    return True, "\n".join(lines)


def main():
    kind, path = sys.argv[1], sys.argv[2]
    body = Path(path).read_text()
    try:
        ok, report = measurement(body) if kind == "measurement" else part(body)
    except (ValueError, json.JSONDecodeError) as e:
        ok, report = False, f"**Rejected.** {e}"
    Path("intake-report.md").write_text(report + "\n")
    print(report)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
