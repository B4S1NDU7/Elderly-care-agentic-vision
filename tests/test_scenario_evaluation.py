import unittest

from scenario_evaluation import evaluate_suite


class ScriptedScenarioEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = evaluate_suite()
        cls.cases = {
            case["name"]: case for case in cls.result["scenarios"]
        }

    def test_suite_reports_scripted_scope_and_all_cases(self):
        self.assertEqual(self.result["evaluation_type"], "scripted_temporal_stress_test")
        self.assertEqual(self.result["aggregate"]["scenario_count"], 6)
        self.assertIn("not image recognition", self.result["important_limit"])
        self.assertEqual(
            self.result["aggregate"]["evaluated_seconds"],
            sum(case["observation_duration_sec"] for case in self.result["scenarios"]),
        )

    def test_brief_stand_does_not_create_exit(self):
        case = self.cases["brief_stand_and_return"]
        self.assertEqual(case["activity_accuracy"], 1.0)
        self.assertEqual(case["bed_exit_metrics"]["false_positives"], 0)
        self.assertEqual(case["bed_return_metrics"]["false_positives"], 0)

    def test_unknown_is_preserved_for_scripted_blanket_occlusion(self):
        case = self.cases["blanket_occlusion_unknown"]
        self.assertEqual(case["activity_accuracy"], 1.0)
        self.assertFalse(case["observed_label_mismatches"])

    def test_scripted_failure_examples_measure_distinct_error_modes(self):
        caregiver = self.cases["caregiver_identity_switch"]
        low_light = self.cases["poor_lighting_forced_posture"]
        camera_loss = self.cases["camera_view_loss_false_absence"]

        self.assertEqual(caregiver["bed_exit_metrics"]["false_positives"], 1)
        self.assertEqual(camera_loss["bed_exit_metrics"]["false_positives"], 1)
        self.assertEqual(low_light["observed_label_mismatches"][0]["expected"], "unknown")
        self.assertEqual(low_light["observed_label_mismatches"][0]["observed"], "lying_in_bed")
        self.assertEqual(caregiver["activity_accuracy"], 0.75)
        self.assertEqual(low_light["activity_accuracy"], 0.75)
        self.assertEqual(camera_loss["activity_accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()
