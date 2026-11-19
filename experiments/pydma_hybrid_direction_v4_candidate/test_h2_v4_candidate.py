import unittest

from run_h2_v4_candidate import _candidate_direction


class H2V4CandidatePolicyTests(unittest.TestCase):
    def test_charge_only_evidence_can_select_charge(self):
        direction, status = _candidate_direction(9, 0)
        self.assertEqual(direction, "charge")
        self.assertEqual(status, "single_direction_charge_only")

    def test_discharge_only_evidence_can_select_discharge(self):
        direction, status = _candidate_direction(0, 4)
        self.assertEqual(direction, "discharge")
        self.assertEqual(status, "single_direction_discharge_only")

    def test_both_unpaired_directions_remain_ambiguous(self):
        direction, status = _candidate_direction(3, 2)
        self.assertEqual(direction, "")
        self.assertEqual(status, "ambiguous_both_directions_without_matched_checkpoints")

    def test_no_accepted_fits_remain_unavailable(self):
        direction, status = _candidate_direction(0, 0)
        self.assertEqual(direction, "")
        self.assertEqual(status, "no_accepted_single_direction_fits")


if __name__ == "__main__":
    unittest.main()
