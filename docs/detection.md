# Detection System

The detection system identifies BFRBs using computer vision and temporal analysis.

## Detection Pipeline

```
Frame Capture (V4L2)
        │
        ▼
┌───────────────────┐
│ Palm Detection    │ → Locate hands in frame
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Hand Landmark     │ → 21 keypoints per hand
└─────────┬─────────┘
          │
          ├─────────────────────────────┐
          │                             │
          ▼                             ▼
┌───────────────────┐         ┌───────────────────┐
│ Face Detection    │         │ Pose Detection    │
└─────────┬─────────┘         └─────────┬─────────┘
          │                             │
          ▼                             ▼
┌───────────────────┐         ┌───────────────────┐
│ Face Mesh         │         │ Pose Landmark     │
│ (468 keypoints)   │         │ (33 keypoints)    │
└─────────┬─────────┘         └─────────┬─────────┘
          │                             │
          └──────────────┬──────────────┘
                         │
                         ▼
              ┌───────────────────┐
              │ Spatial Analysis  │
              │ (distances, etc.) │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Behavior Detector │
              │ (per BFRB type)   │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Temporal Tracker  │
              │ (sliding window)  │
              └─────────┬─────────┘
                        │
                        ▼
                   Detection Event
```

## ONNX Models

| Model | Input | Output | Source |
|-------|-------|--------|--------|
| Palm Detection | 192x192 RGB | Bounding boxes | MediaPipe |
| Hand Landmark | 224x224 RGB | 21 keypoints | MediaPipe |
| Face Detection | 128x128 RGB | Bounding boxes | BlazeFace |
| Face Mesh | 192x192 RGB | 468 keypoints | MediaPipe |
| Pose Detection | 224x224 RGB | Bounding box | BlazePose |
| Pose Landmark | 256x256 RGB | 33 keypoints | BlazePose |

Models are downloaded automatically on first run from:
- [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo)
- [HuggingFace](https://huggingface.co/)

## Hand Landmarks

21 keypoints per hand:

```
        8   12  16  20
        |   |   |   |
    7   11  15  19  |
    |   |   |   |   |
    6   10  14  18  |
    |   |   |   |   |
    5---9---13--17--+
         \         |
          4        |
           \       |
            3      |
             \     |
              2    |
               \   |
                1  |
                 \ |
                  0 (wrist)
```

Key indices:
- **Fingertips:** 4, 8, 12, 16, 20
- **Wrist:** 0
- **Thumb tip:** 4
- **Index tip:** 8

## Face Mesh Landmarks

468 keypoints covering the face. Key regions:

- **Lips:** Indices for upper/lower lip contours
- **Mouth center:** Average of lip landmarks
- **Nose tip:** Index 1
- **Chin:** Lower contour indices

## Behavior Detectors

### Nail Biting

Detects when fingertips are near the mouth:

```rust
fn analyze(&self, ctx: &DetectionContext) -> Option<AnalysisResult> {
    let mouth_center = ctx.analyzer.mouth_center()?;
    let face_width = ctx.analyzer.face_width()?;

    for hand in &ctx.hands {
        for &tip_idx in &FINGERTIP_INDICES {
            let tip = hand.landmarks[tip_idx];
            let distance = distance_2d(tip, mouth_center) / face_width;

            if distance < self.proximity_threshold {
                return Some(AnalysisResult {
                    bfrb_type: BfrbType::NailBiting,
                    confidence: 1.0 - distance / self.proximity_threshold,
                    ..
                });
            }
        }
    }
    None
}
```

**Configuration:**
```yaml
nail_biting:
  proximity_threshold: 0.35  # Normalized by face width
  min_sustained_ms: 1500
  confidence_threshold: 0.3
```

### Nail Picking

Detects when fingertips of both hands are close together:

```rust
fn analyze(&self, ctx: &DetectionContext) -> Option<AnalysisResult> {
    if ctx.hands.len() < 2 {
        return None;
    }

    let hand1 = &ctx.hands[0];
    let hand2 = &ctx.hands[1];

    for &tip1 in &FINGERTIP_INDICES {
        for &tip2 in &FINGERTIP_INDICES {
            let distance = distance_2d(
                hand1.landmarks[tip1],
                hand2.landmarks[tip2]
            );

            if distance < self.proximity_threshold {
                return Some(AnalysisResult { .. });
            }
        }
    }
    None
}
```

## Temporal Tracking

Raw detections are noisy. The temporal tracker confirms sustained behavior:

```
Confidence over time:

1.0 ─┐     ╭──╮    ╭────────────╮
     │    ╱    ╲  ╱              ╲
0.5 ─┼───╱──────╲╱────────────────╲───
     │  ╱                          ╲
0.0 ─┴─╱────────────────────────────╲─

     │←─ Accumulating ─→│←─ Confirmed ─→│
```

### Tracking Phases

1. **Idle** - No detection
2. **Accumulating** - Detection started, gathering evidence
3. **Confirmed** - Sustained detection confirmed → trigger alert
4. **Alerting** - Alert active, waiting for resolution
5. **Cooldown** - Post-exercise cooldown period

### Sliding Window

```yaml
detection:
  temporal:
    window_ms: 1500       # 1.5 second window
    positive_ratio: 0.4   # 40% of frames must be positive
```

At 8 FPS with 1500ms window:
- Window holds ~12 frames
- Need 5+ positive frames for confirmation
- Handles intermittent palm detection failures

## False Positive Suppression

### Typing Suppression

Detects rapid alternating hand movements characteristic of typing:
- Hands moving in opposite directions
- High movement frequency
- Hands near keyboard level

### Chin Rest Suppression

Detects hand resting on chin (not biting):
- Hand stationary for extended period
- Palm facing down
- No fingertip-to-mouth proximity

### Eating Suppression

Detects eating gestures:
- Regular hand-to-mouth movements
- Object (utensil) detected between fingers
- Consistent mouth opening

## Multi-Camera Fusion

When multiple cameras are configured:

```yaml
fusion:
  strategy: any           # any, all, merge
  merge_tolerance_ms: 100
```

- **any** - Detect if any camera sees behavior
- **all** - Require all cameras to agree
- **merge** - Combine detections from cameras within tolerance

## Tuning Detection

### Too Many False Positives

1. Increase proximity threshold:
   ```yaml
   proximity_threshold: 0.25  # Stricter (was 0.35)
   ```

2. Increase positive ratio:
   ```yaml
   positive_ratio: 0.6  # Need more evidence (was 0.4)
   ```

3. Enable suppression:
   ```yaml
   false_positive:
     chin_rest_suppression: true
     eating_suppression: true
   ```

### Too Many False Negatives

1. Decrease proximity threshold:
   ```yaml
   proximity_threshold: 0.45  # More lenient
   ```

2. Decrease positive ratio:
   ```yaml
   positive_ratio: 0.3  # Less evidence needed
   ```

3. Increase window size:
   ```yaml
   window_ms: 2000  # Longer window
   ```

## Debug Logging

Enable detailed detection logs:

```bash
RUST_LOG=nailbite::detection=debug pnpm tauri dev
```

Output includes:
- Per-frame confidence values
- Spatial distances
- Tracker state transitions
- Suppression activations
