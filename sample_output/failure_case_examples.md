SCRIPTED TEMPORAL STRESS EVALUATION
========================================================================
Method: scripted state observations are supplied to the temporal tracker.

Scenarios: 10
Total scripted observation: 821s
Weighted state accuracy: 98.2%
Bed exits: TP=1 FP=2 FN=0
Bed returns: TP=1 FP=2 FN=0

Scenario                            Accuracy  Duration MAE  Exit FP  Return FP
--------------------------------------------------------------------------------
routine_states_and_bed_events        100.0%         0.0s        0          0
brief_stand_and_return               100.0%         0.0s        0          0
turning_while_lying                  100.0%         0.0s        0          0
prolonged_edge_sitting_monitor       100.0%         0.0s        0          0
temporary_occlusion_unknown          100.0%         0.0s        0          0
caregiver_enters_resident_stays_in_bed   100.0%         0.0s        0          0
blanket_occlusion_unknown            100.0%         0.0s        0          0
caregiver_identity_switch             75.0%         3.3s        1          1
poor_lighting_forced_posture          75.0%         5.0s        0          0
camera_view_loss_false_absence        75.0%         3.3s        1          1

ALERT RULE CHECKS
--------------------------------------------------------------------------------
routine_activity                   expected=NORMAL  actual=NORMAL  PASS
prolonged_sitting_on_bed           expected=MONITOR actual=MONITOR PASS
prolonged_unknown                  expected=MONITOR actual=MONITOR PASS
prolonged_absence_monitor          expected=MONITOR actual=MONITOR PASS
prolonged_absence_alert            expected=ALERT   actual=ALERT   PASS
suspected_floor_fall               expected=ALERT   actual=ALERT   PASS

SCRIPTED FAILURE EXAMPLES
--------------------------------------------------------------------------------
- caregiver_identity_switch: Injected classifier failure: during resident occlusion, a caregiver is mistaken for the resident.
  Accuracy=75.0%, duration MAE=3.3s, exit FP=1, return FP=1.
  Example at 5s: expected unknown, observed sitting_outside_bed.
- poor_lighting_forced_posture: Injected classifier failure: uncertain dark frames are forced to LYING_IN_BED instead of UNKNOWN.
  Accuracy=75.0%, duration MAE=5.0s, exit FP=0, return FP=0.
  Example at 5s: expected unknown, observed lying_in_bed.
- camera_view_loss_false_absence: Injected classifier failure: temporary loss of view is misread as OUT_OF_BED.
  Accuracy=75.0%, duration MAE=3.3s, exit FP=1, return FP=1.
  Example at 5s: expected unknown, observed out_of_bed.
