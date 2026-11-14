"""跨夜通量校正的回归测试。

运行:python -m pytest tests/ -q
"""
import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import calibration

SAMPLE = Path(__file__).resolve().parent.parent / "sample"


def read(name: str) -> bytes:
    return (SAMPLE / name).read_bytes()


def ref_csv(rows: list[tuple[str, str, float]]) -> bytes:
    text = "night_id,star_id,flux\n" + "".join(
        f"{n},{s},{f}\n" for n, s, f in rows)
    return text.encode()


def target_csv(rows: list[tuple[str, float, float]]) -> bytes:
    text = "night_id,time_day,flux\n" + "".join(
        f"{n},{t},{f}\n" for n, t, f in rows)
    return text.encode()


def test_bridge_sample_calibrates_every_night():
    report = calibration.run_calibration(
        read("reference_bridge.csv"), read("target_star.csv"),
        "S_ANCHOR", 100.0, 0.05)
    got = {n["night_id"]: n["throughput"] for n in report["nights"]}
    expected = {"N1": 0.90, "N2": 0.80, "N3": 1.10,
                "N4": 0.95, "N5": 1.05, "N6": 0.85}
    assert got.keys() == expected.keys()
    for night, value in expected.items():
        assert got[night] == pytest.approx(value, rel=1e-6)
    # 第 7 夜不在参照图中 → 未校准,其目标观测被舍弃
    assert report["uncalibrated_nights"] == [
        {"night_id": "N7", "n_target_rows": 14,
         "reason": "not_in_reference"}]
    assert report["n_target_calibrated"] + report["n_target_excluded"] \
        == report["n_target_rows"]


def test_output_is_sorted_and_on_one_scale():
    report = calibration.run_calibration(
        read("reference_bridge.csv"), read("target_star.csv"),
        "S_ANCHOR", 100.0, 0.05)
    df = pd.read_csv(io.StringIO(report["csv"]))
    assert list(df.columns) == ["time_day", "flux"]
    assert df["time_day"].is_monotonic_increasing
    assert (df["flux"] > 0).all()
    # 目标星真实通量在 7.6~8.4 之间,校正后应当还原到该尺度
    assert df["flux"].min() > 7.0 and df["flux"].max() < 9.0


def test_conflict_sample_lists_nights_and_stars():
    with pytest.raises(calibration.CalibrationError) as info:
        calibration.run_calibration(
            read("reference_conflict.csv"), read("target_star.csv"),
            "S_ANCHOR", 100.0, 0.05)
    detail = info.value.detail
    assert detail["code"] == "reference_conflict"
    assert detail["conflict_nights"] == ["N3", "N5"]
    assert detail["conflict_stars"] == ["S_ANCHOR", "S_BRIDGE"]
    pairs = {(c["night_id"], c["star_id"]) for c in detail["conflicts"]}
    assert pairs == {("N3", "S_ANCHOR"), ("N5", "S_BRIDGE")}


def test_one_bad_measurement_is_pinpointed():
    rows = [("N1", "A", 90.0), ("N1", "B", 54.0),
            ("N2", "A", 80.0), ("N2", "C", 32.0),
            ("N3", "A", 110.0), ("N3", "C", 44.0),
            ("N4", "B", 57.0), ("N4", "C", 38.0)]
    rows = [(n, s, f * (1.2 if (n, s) == ("N3", "C") else 1.0))
            for n, s, f in rows]
    with pytest.raises(calibration.CalibrationError) as info:
        calibration.run_calibration(
            ref_csv(rows), read("target_star.csv"), "A", 100.0, 0.05)
    detail = info.value.detail
    assert detail["conflict_nights"] == ["N3"]
    assert detail["conflict_stars"] == ["C"]


def test_disconnected_nights_are_uncalibrated():
    rows = [("N1", "S_ANCHOR", 90.0), ("N1", "S_BRIDGE", 54.0),
            ("N2", "S_ANCHOR", 80.0), ("N2", "S_BRIDGE", 48.0),
            ("N4", "S_X", 44.0), ("N4", "S_Y", 33.0),
            ("N5", "S_Y", 46.0)]
    report = calibration.run_calibration(
        ref_csv(rows), read("target_star.csv"), "S_ANCHOR", 100.0, 0.05)
    assert [n["night_id"] for n in report["nights"]] == ["N1", "N2"]
    reasons = {u["night_id"]: u["reason"]
               for u in report["uncalibrated_nights"]}
    assert reasons["N4"] == "disconnected_from_anchor"
    assert reasons["N5"] == "disconnected_from_anchor"
    assert reasons["N3"] == "not_in_reference"
    df = pd.read_csv(io.StringIO(report["csv"]))
    # N4/N5/N3/N7 的目标观测都不在结果里
    assert len(df) == report["n_target_calibrated"]


def test_target_too_short_after_filtering_is_rejected():
    target = target_csv([("N1", i * 0.1, 8.0 + i * 0.01) for i in range(5)] +
                        [("N7", 5.0 + i * 0.1, 8.0) for i in range(20)])
    with pytest.raises(calibration.CalibrationError) as info:
        calibration.run_calibration(
            read("reference_bridge.csv"), target, "S_ANCHOR", 100.0, 0.05)
    assert info.value.detail["code"] == "target_entry_rules"
    assert info.value.detail["n_remaining"] == 5
    assert "10" in info.value.detail["message"]


def test_duplicate_target_time_is_rejected():
    rows = [("N1", float(i), 8.0) for i in range(10)]
    rows.append(("N1", 3.0, 8.2))
    with pytest.raises(calibration.CalibrationError) as info:
        calibration.run_calibration(
            read("reference_bridge.csv"), target_csv(rows),
            "S_ANCHOR", 100.0, 0.05)
    assert info.value.detail["code"] == "target_entry_rules"
    assert info.value.detail["duplicate_times"] == [3.0]


@pytest.mark.parametrize("anchor_flux,max_dev", [
    (0.0, 0.05), (-1.0, 0.05), (float("nan"), 0.05), (100.0, -0.1),
    (100.0, float("inf")),
])
def test_bad_parameters_rejected(anchor_flux, max_dev):
    with pytest.raises(calibration.CalibrationError):
        calibration.run_calibration(
            read("reference_bridge.csv"), read("target_star.csv"),
            "S_ANCHOR", anchor_flux, max_dev)


def test_nonpositive_flux_rejected():
    target = target_csv([("N1", float(i), 8.0) for i in range(10)])
    target += b"N1,20.0,-3.0\n"
    with pytest.raises(calibration.CalibrationError) as info:
        calibration.run_calibration(
            read("reference_bridge.csv"), target, "S_ANCHOR", 100.0, 0.05)
    assert info.value.detail["code"].endswith("_nonpositive")


def test_unknown_anchor_rejected():
    with pytest.raises(calibration.CalibrationError) as info:
        calibration.run_calibration(
            read("reference_bridge.csv"), read("target_star.csv"),
            "NO_SUCH_STAR", 100.0, 0.05)
    assert info.value.detail["code"] == "anchor_missing"
    assert "S_ANCHOR" in info.value.detail["known_stars"]
