import pytest

from bafuse.features.physics_coverage import FEATURES, nominal_age, pulse_droop, raw_droop, rest_and_pulse


def test_cycle_age_uses_continuing_guide_ordinal():
    row = {"dataset_part": "2", "group": "1", "rpt_index": "13",
           "physical_cell_id": "G1-Cell15", "guide_stage": "RPT13", "source_tpg": "TPG1.22"}
    mapping = {"procedure": "TPG1.22", "cycle_age_status": "nominal_48_cycles_per_RPT",
               "chronology_basis": "guide_rpt_ordinal", "nominal_cycle_age": "624"}
    assert nominal_age(row, mapping)[0] == 624
    with pytest.raises(ValueError, match="Guide stage"):
        nominal_age({**row, "guide_stage": "Profile + RPT1–RPT4"}, mapping)


@pytest.mark.parametrize(("part", "steps"), [(1, (21, 22)), (2, (21, 22)), (3, (19, 20))])
def test_nominal_mid_soc_pulse_steps(part, steps):
    assert rest_and_pulse(part) == steps


def test_droop_interpolates_at_ten_seconds_instead_of_end_of_pulse():
    rest = [(0.0, 0.0, 0.0, 3.8)]
    pulse = [(0.001 + s / 60, s / 60, -1.5, 3.7 - 0.01 * s)
             for s in [0, 9.5, 10.5, 36]]
    drop, duration, left, right = pulse_droop(rest, pulse)
    assert drop == pytest.approx(0.2)
    assert (duration, left, right) == pytest.approx((36, 9.5, 10.5))
    with pytest.raises(ValueError, match="does not bracket"):
        pulse_droop(rest, pulse[:2])


def test_updated_physics_feature_order_excludes_current_descriptors():
    assert FEATURES[:2] == ("nominal_cycle_age", "voltage_droop_50_soc")
    assert len(FEATURES) == 5


def test_raw_droop_stays_within_the_frozen_rpt(tmp_path):
    directory = tmp_path / "data/raw/oxford"
    directory.mkdir(parents=True)
    source = directory / "test.csv"
    header = "Step,TestTime,StepTime,Amps,Volts\n"
    charge = "10,0,0,0.1,3.7\n" * 7
    rest = "21,1,0,0,3.8\n"
    pulse = "".join(f"22,{1.001+s/60},{s/60},-1.5,{3.7-s/100}\n" for s in (0, 9.5, 10.5))
    row = {"dataset_part": "1", "source_file": source.relative_to(tmp_path).as_posix(),
           "step10_segment_reference": "test.csv:rows2-8"}
    source.write_text(header + charge + rest + pulse, encoding="utf-8")
    assert raw_droop(tmp_path, row)[0] == pytest.approx(0.2)
    source.write_text(header + charge + "4,0,0,-1.5,3.7\n" + rest + pulse, encoding="utf-8")
    with pytest.raises(ValueError, match="Missing adjacent rest or pulse"):
        raw_droop(tmp_path, row)
