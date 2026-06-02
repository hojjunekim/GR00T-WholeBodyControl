"""Default initial poses for VLA inference.

These arrays are sent to the C++ control loop when the user presses 'i' or 'o'
to move the robot to a known starting configuration before inference begins.

WARNING: These tokens are specific to the SONIC checkpoint used during training.
Different SONIC checkpoints encode different latent spaces. When changing the
SONIC checkpoint, update both tokens accordingly.

  i -> LATENT_IDEAL_POSE:  neutral standing (average over all training frames)
  o -> LATENT_START_POSE:  task-ready pose, arms at 90 degrees
                           (average over first frame of each training episode)
"""

import numpy as np

# Neutral standing pose — captured live from the PLANNER's token_state while
# the robot was standing idle in sim (after pressing k, before any task).
LATENT_IDEAL_POSE = np.array(
    [
        -0.0625,   0.1875,   0.0625,  -0.0625,   0.0625,  -0.1875,   0.3750,
         0.1250,   0.1250,   0.0625,  -0.1250,  -0.2500,  -0.1250,   0.0000,
        -0.1875,  -0.0625,   0.1250,  -0.0625,  -0.1875,  -0.3125,   0.1875,
         0.0625,   0.0625,  -0.2500,  -0.0625,   0.0000,   0.0625,  -0.4375,
         0.0000,   0.0625,   0.1250,  -0.1250,   0.3125,  -0.0625,  -0.1250,
        -0.2500,  -0.0625,  -0.1250,   0.1875,  -0.0625,   0.0000,  -0.3125,
        -0.0625,   0.0000,  -0.2500,   0.0000,   0.0000,  -0.1250,  -0.0625,
         0.1250,  -0.1250,   0.0625,   0.1875,   0.1875,   0.0625,   0.0000,
         0.0000,   0.3750,   0.0000,  -0.4375,   0.2500,   0.0000,   0.0625,
        -0.0625,
    ],
    dtype=np.float32,
)

# Task-ready pose — average over the FIRST frame of each training episode.
# Arms bent ~90 degrees, the pose used at the start of every demonstration.
LATENT_START_POSE = np.array(
    [
        -0.1250,  0.0000, -0.1250,  0.0000,  0.0000,  0.0625, -0.0625,
         0.0625,  0.0000, -0.0625,  0.0625,  0.1250, -0.1875, -0.1875,
        -0.1875,  0.0000, -0.0625, -0.0625, -0.1250, -0.1250, -0.1250,
        -0.0625,  0.0625, -0.1875, -0.0625,  0.1250,  0.0625,  0.1250,
         0.1250,  0.1250,  0.1875, -0.1250, -0.1875,  0.1250, -0.1250,
         0.0000, -0.0625, -0.1250,  0.1250,  0.0625, -0.0625,  0.0000,
        -0.1250,  0.0000, -0.0625,  0.0000,  0.1875, -0.1875,  0.0625,
         0.2500, -0.0625,  0.0625,  0.0625, -0.0625,  0.0625, -0.0625,
         0.0000,  0.1875,  0.0625,  0.1250,  0.0000,  0.0000,  0.1250,
         0.0000,
    ],
    dtype=np.float32,
)

# Keep backward-compatible alias
LATENT_INITIAL_MOTION_TOKEN = LATENT_IDEAL_POSE
