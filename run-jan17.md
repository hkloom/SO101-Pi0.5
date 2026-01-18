============================================================
Pi0.5 Inference for SO-101
============================================================
Loading policy from ./pi05_fixed...
⚠️  DISCLAIMER: The PI05OpenPI model is a direct PyTorch port of the OpenPI implementation. 
   This implementation follows the original OpenPI structure for compatibility. 
   Original implementation: https://github.com/Physical-Intelligence/openpi
Loading model from: ./pi05_fixed
Could not load state dict from remote files: Error while deserializing header: incomplete metadata, file not fully covered
Returning model without loading pretrained weights
  Policy type: pi05_openpi
  Device: cuda
  ⚠️  WARNING: No unnormalization stats loaded!
     Actions will NOT be denormalized properly.
     Make sure model.safetensors contains 'unnormalize_outputs.*' keys.
  ⚠️  WARNING: No normalization stats loaded for inputs!

Loading postprocessor...
  ✓ Loaded postprocessor from ./pi05_fixed
    Stats loaded for: ['action', 'episode_index', 'frame_index', 'index', 'observation.images.main', 'observation.images.secondary_0', 'observation.images.secondary_1', 'observation.state', 'task_index', 'timestamp']
    Action stats keys: ['count', 'max', 'mean', 'min', 'q01', 'q10', 'q50', 'q90', 'q99', 'std']
      count: [19597.]
      max: [0.4817876  0.6781851  1.4131414  1.8212799  0.12888585 0.88072   ]
      mean: [ 0.0472376  -0.889073    0.6359377   1.2380551  -0.10721558  0.08579539]
      min: [-0.185657   -1.8028675  -0.49406242  0.01227484 -0.50326854 -0.05063373]
      q01: [-0.05285679 -1.7823659  -0.1543853   0.9357583  -0.19229412 -0.02464019]
      q10: [-0.03022041 -1.7812798  -0.13597444  1.0260501  -0.18824725 -0.02404539]
      q50: [ 0.02423145 -1.2907145   0.59193695  1.2888073  -0.11354612 -0.00974369]
      q90: [ 0.1521697   0.3517829   1.3954235   1.3861498  -0.02420907  0.58738595]
      q99: [ 0.16882905  0.48180854  1.3962821   1.5184761  -0.00918255  0.6547676 ]
      std: [0.09820139 0.9493389  0.7037728  0.18368992 0.11021108 0.23403607]

Opening cameras...
  Opened camera: /dev/video4
  Opened camera: /dev/video2
  Opened camera: /dev/video0

Connecting to robot on /dev/ttyACM0...
  Robot connected!
  Motors: ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
Policy reset.

============================================================
Starting inference loop
  Task: Pick up the red lego.
  Duration: 60.0s
  Target FPS: 30.0
  Async inference: disabled
  Action unnormalization: ✓ Using postprocessor
  Press Ctrl+C to stop
============================================================

Initial robot state: [ 3.6778226e-02 -9.9079498e+01  1.0000000e+02  7.4772430e+01
  7.0573869e+00  2.1293800e+01]

[CHUNK] Generated 50 actions in 0.85s

--- Step 0 ---
  Unnormalization: postprocessor (QUANTILES [-1,1]->rad->deg)
  Raw action: [-0.66603154 -1.3236357  -0.12082896 -2.9127467   0.16114667 -0.71206343]
  Processed action: [  -0.9074938 -123.11434     30.210108    21.68431     -4.926546
    4.192505 ]
CALLING execute_action
INSIDE execute_action
action_dict:  {'shoulder_pan.pos': -0.9074938297271729, 'shoulder_lift.pos': -123.11434173583984, 'elbow_flex.pos': 30.210107803344727, 'wrist_flex.pos': 21.684310913085938, 'wrist_roll.pos': -4.926546096801758, 'gripper.pos': 4.1925048828125}
  Current state: [ 3.6778226e-02 -9.9079498e+01  1.0000000e+02  7.4772430e+01
  7.0573869e+00  2.1293800e+01]
  Requested action: [  -0.9074938 -123.11434     30.210108    21.68431     -4.926546
    4.192505 ]
  Diff: [ -0.94427204 -24.034843   -69.789894   -53.08812    -11.9839325
 -17.101295  ]
CALLING send_action
INSIDE send_action
send_action 1
send_action 2
send_action 2.1
send_action 2.2
send_action 2.3
WARNING:root:Relative goal position magnitude had to be clamped to be safe.
{   'elbow_flex': {   'original goal_pos': 30.210107803344727,
                      'safe goal_pos': 99.95},
    'gripper': {   'original goal_pos': 4.1925048828125,
                   'safe goal_pos': 21.243800539083555},
    'shoulder_lift': {   'original goal_pos': -123.11434173583984,
                         'safe goal_pos': -99.12949790794978},
    'shoulder_pan': {   'original goal_pos': -0.9074938297271729,
                        'safe goal_pos': -0.013221772710554094},
    'wrist_flex': {   'original goal_pos': 21.684310913085938,
                      'safe goal_pos': 74.72243172951887},
    'wrist_roll': {   'original goal_pos': -4.926546096801758,
                      'safe goal_pos': 7.007387057387061}}
send_action 3
send_action 4
send_action returned
  Sent action: [-0.013221772710554094, -99.12949790794978, 99.95, 74.72243172951887, 7.007387057387061, 21.243800539083555]

--- Step 1 ---
  Unnormalization: postprocessor (QUANTILES [-1,1]->rad->deg)
  Raw action: [-7.6798481e-01 -3.0938387e-03 -5.1763535e-02  3.5577331e+00
  1.9547973e+00  1.6996523e+00]
  Processed action: [ -1.554982 -37.458916  33.278225 129.70018    4.482518  51.133186]
CALLING execute_action
INSIDE execute_action
action_dict:  {'shoulder_pan.pos': -1.5549819469451904, 'shoulder_lift.pos': -37.45891571044922, 'elbow_flex.pos': 33.27822494506836, 'wrist_flex.pos': 129.70018005371094, 'wrist_roll.pos': 4.482518196105957, 'gripper.pos': 51.13318634033203}
  Current state: [ 3.6778226e-02 -9.9079498e+01  1.0000000e+02  7.4772430e+01
  7.0573869e+00  2.1293800e+01]
  Requested action: [ -1.554982 -37.458916  33.278225 129.70018    4.482518  51.133186]
  Diff: [ -1.5917602  61.620583  -66.72177    54.92775    -2.5748687  29.839386 ]
CALLING send_action
INSIDE send_action
send_action 1
send_action 2
send_action 2.1
send_action 2.2
send_action 2.3
WARNING:root:Relative goal position magnitude had to be clamped to be safe.
{   'elbow_flex': {   'original goal_pos': 33.27822494506836,
                      'safe goal_pos': 99.95},
    'gripper': {   'original goal_pos': 51.13318634033203,
                   'safe goal_pos': 21.343800539083556},
    'shoulder_lift': {   'original goal_pos': -37.45891571044922,
                         'safe goal_pos': -99.02949790794979},
    'shoulder_pan': {   'original goal_pos': -1.5549819469451904,
                        'safe goal_pos': -0.013221772710554094},
    'wrist_flex': {   'original goal_pos': 129.70018005371094,
                      'safe goal_pos': 74.82243172951887},
    'wrist_roll': {   'original goal_pos': 4.482518196105957,
                      'safe goal_pos': 7.007387057387061}}
send_action 3
send_action 4
send_action returned
  Sent action: [-0.013221772710554094, -99.02949790794979, 99.95, 74.82243172951887, 7.007387057387061, 21.343800539083556]
