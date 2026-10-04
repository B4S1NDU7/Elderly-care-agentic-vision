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
        self.assertGreaterEqual(self.result["aggregate"]["scenario_count"], 10)
        self.assertIn("evaluation_scope", self.result)
        self.assertEqual(
            self.result["aggregate"]["evaluated_seconds"],
            sum(case["observation_duration_sec"] for case in self.result["scenarios"]),
        )

    def test_all_required_difficult_situations_have_scenarios(self):
        self.assertTrue({
            "turning_while_lying",
            "brief_stand_and_return",
            "prolonged_edge_sitting_monitor",
            "routine_states_and_bed_events",
            "blanket_occlusion_unknown",
            "temporary_occlusion_unknown",
            "caregiver_enters_resident_stays_in_bed",
            "caregiver_identity_switch",
            "poor_lighting_forced_posture",
            "camera_view_loss_false_absence",
        }.issubset(self.cases))
        routine = self.cases["routine_states_and_bed_events"]
        self.assertTrue({
            "sitting_outside_bed",
            "walking",
            "out_of_bed",
            "sitting_on_bed",
        }.issubset({
            segment["state"] for segment in routine["predicted_timeline"]
        }))

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

    def test_alert_rules_cover_normal_monitor_and_alert(self):
        checks = self.result["alert_rule_checks"]
        self.assertEqual(len(checks), 6)
        self.assertTrue(all(check["passed"] for check in checks))
        self.assertEqual(
            {check["actual"] for check in checks},
            {"NORMAL", "MONITOR", "ALERT"},
        )


if __name__ == "__main__":
    unittest.main()
