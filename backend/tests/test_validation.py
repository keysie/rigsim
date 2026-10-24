"""Guard rail: the model must stay close to the measured points (see validation.py)."""
from tests import validation


def test_calibration_and_validation_points():
    validation.POINTS.clear()
    validation.run()
    for grp, within, med in (("CAL", 16, 0.2), ("VAL", 34, 0.25)):
        rows = [p for p in validation.POINTS if p[0] == grp]
        errs = sorted(abs(m - t) / t for _, _, m, t, _ in rows)
        assert sum(e <= 0.3 for e in errs) >= within, f"{grp}: only {sum(e <= 0.3 for e in errs)} of {len(errs)} within 30 %"
        assert errs[len(errs) // 2] <= med, f"{grp}: median error {errs[len(errs) // 2]:.0%}"
