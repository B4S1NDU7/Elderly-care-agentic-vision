SCRIPTED TEMPORAL STRESS EVALUATION
========================================================================
LIMIT: Scripted state labels are fed directly to the temporal tracker.
This is not a visual-model evaluation and is not real-video evidence.

Scenarios: 6
Total scripted observation: 164s
Weighted state accuracy: 90.9%
Bed exits: TP=1 FP=2 FN=0
Bed returns: TP=1 FP=2 FN=0

Scenario                            Accuracy  Duration MAE  Exit FP  Return FP
--------------------------------------------------------------------------------
routine_states_and_bed_events        100.0%         0.0s        0          0
brief_stand_and_return               100.0%         0.0s        0          0
blanket_occlusion_unknown            100.0%         0.0s        0          0
caregiver_identity_switch             75.0%         3.3s        1          1
poor_lighting_forced_posture          75.0%         5.0s        0          0
camera_view_loss_false_absence        75.0%         3.3s        1          1

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
