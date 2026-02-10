# Exercises

Decoupling exercises help break the automatic connection between urge and behavior. When a BFRB is detected, an exercise is presented to redirect the behavior.

## Available Exercises

### Fist Clench

**Category:** Timed Hold (60 seconds)
**Applicable to:** Nail Biting, Nail Picking

Make tight fists with both hands and hold. This engages the same muscles used in the BFRB but in a controlled, deliberate way.

**Verification:** Both hands detected with fingers curled (fingertip-to-palm distance < threshold).

### Palm Press

**Category:** Timed Hold (45 seconds)
**Applicable to:** Nail Biting, Nail Picking, Hair Pulling, Skin Picking

Press palms together in front of chest. Creates isometric tension in hands and arms.

**Verification:** Both hands detected, palms facing each other, close proximity.

### Flat Hand Press

**Category:** Timed Hold (30 seconds)
**Applicable to:** Nail Biting, Nail Picking

Place both hands flat on a surface (desk, thighs). Prevents hand-to-face contact.

**Verification:** Both hands detected with fingers extended (flat hand posture).

### Fingertip Massage

**Category:** Timed Hold (30 seconds)
**Applicable to:** Nail Picking

Gently massage fingertips of one hand with the other. Provides tactile stimulation without damage.

**Verification:** Hands in close proximity with gentle contact detected.

### Interlocked Squeeze

**Category:** Timed Hold (45 seconds)
**Applicable to:** Nail Biting, Nail Picking

Interlock fingers and squeeze hands together. Strong engagement of hand muscles.

**Verification:** Hands overlapping with interlocked finger pattern.

### Ear Touch

**Category:** Repetitions (10 reps)
**Applicable to:** Nail Biting

Touch earlobes alternately. Redirects hand-to-face movement to a harmless target.

**Verification:** Hand detected near ear region of face mesh.

### Finger Flick

**Category:** Repetitions (20 reps)
**Applicable to:** Nail Picking

Flick fingers outward rapidly. Releases tension and provides sensory input.

**Verification:** Rapid finger extension detected.

## Exercise Selection

Configured in `config.yaml`:

```yaml
exercises:
  selection_strategy: random    # random, round_robin, preferred
  preferred_exercise: null      # Exercise ID for 'preferred' strategy
  hold_duration_override: null  # Override default durations
  reps_override: null           # Override default rep counts
  timeout_seconds: 120          # Max time before timeout
  compliance_ratio: 0.8         # Required pose accuracy (0-1)
```

### Selection Strategies

- **random** - Random exercise applicable to the detected BFRB
- **round_robin** - Cycle through applicable exercises in order
- **preferred** - Always use the specified exercise (if applicable)

## Exercise Flow

```
Detection Confirmed
        │
        ▼
┌───────────────┐
│ Select        │
│ Exercise      │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Show          │
│ Instructions  │
└───────┬───────┘
        │
        ▼
┌───────────────┐     Timeout
│ Active        │─────────────► Exercise Failed
│ Verification  │
└───────┬───────┘
        │ Completed
        ▼
┌───────────────┐
│ Cooldown      │
│ Period        │
└───────────────┘
```

## Verification System

Each exercise has a verification function that receives:
- Current hand landmarks
- Current face landmarks
- Current pose landmarks

And returns:
- `pose_correct: bool` - Whether the pose matches
- `progress: float` - Completion progress (0-1)
- `feedback: string` - User feedback message

### Timed Hold Exercises

Progress increases while pose is correct:
- Progress = time_in_pose / hold_duration
- Completes when progress reaches 1.0
- Resets if pose breaks

### Repetition Exercises

Progress increases on rep completion:
- Progress = completed_reps / target_reps
- Rep counted on correct pose transition
- Debounced to prevent double-counting

## Compliance Ratio

The `compliance_ratio` setting (default 0.8) determines how strict pose verification is:

- **1.0** - Perfect pose required at all times
- **0.8** - 80% of verification checks must pass (recommended)
- **0.5** - Half of checks must pass (lenient)

Lower values are more forgiving of brief pose breaks.

## Adding Custom Exercises

See [Development Guide](development.md#adding-a-new-exercise) for implementation details.

Exercise requirements:
1. Clear, simple instructions
2. Verifiable via hand/face/pose landmarks
3. Applicable to one or more BFRB types
4. Achievable in under 2 minutes
