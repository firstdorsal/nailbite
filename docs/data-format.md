# Event data format

This document describes what Nailbite captures to disk for each detection event.
It exists to make the recorded data legible to anyone interested in training a
proper end-to-end model on it, rather than reverse-engineering the format from
the source.

Nothing here ever leaves your machine. Everything described below lives under
`~/.local/share/nailbite/history/` (overridable via `history.dir` in
`config.yaml`).

## On-disk layout

Each event is one directory.

```
~/.local/share/nailbite/history/
└── 20260518_193045_nail_biting/
    ├── event.json
    └── frames/
        ├── frame_-05.jpg              # 5 frames before the trigger
        ├── frame_-05_annotated.jpg    # same frame with landmark overlay baked in
        ├── frame_-04.jpg
        ├── frame_-04_annotated.jpg
        ├── …
        ├── frame_+00.jpg              # the trigger frame
        ├── frame_+00_annotated.jpg
        ├── frame_+01.jpg
        ├── …
        └── frame_+05.jpg              # 5 frames after the trigger
```

- Directory name: `YYYYMMDD_HHMMSS_<bfrb_type>` (local time).
- `frames_before` and `frames_after` default to 5 each
  (`history.frames_before`, `history.frames_after`); a typical clip is 11
  frames at the inference rate (8 FPS by default, so ~1.4 s of context around
  the trigger).
- `*_annotated.jpg` is the same frame with the same landmarks rendered on
  top — provided only when `history.annotate_landmarks` is on (default).
  For training you want the raw `frame_*.jpg` and the vector overlay in
  `event.json`, not the annotated JPEG.
- Raw frame resolution matches `camera.sources[].resolution_width/height`
  (default 640×480, JPEG-encoded).
- The pre-trigger ring buffer plus post-trigger collection is implemented in
  `src-tauri/src/stats/event_history.rs`.

## `event.json`

Pretty-printed JSON. Skeleton:

```json
{
  "timestamp": "2026-05-18T17:30:45.123Z",
  "trigger": "detection",
  "bfrb_type": "nail_biting",
  "confidence": 0.81,
  "explanation": { /* DetectionExplanation, trigger frame only */ },
  "frames": [
    {
      "offset": -5,
      "timestamp_ms": 123450,
      "hand_count": 2,
      "hand_sides": ["left", "right"],
      "hand_confidences": [0.92, 0.88],
      "face_detected": true,
      "pose_detected": true,
      "overlay": {
        "hands": [ /* HandResult[] */ ],
        "face": { /* FaceResult */ },
        "pose": { /* PoseResult */ }
      },
      "explanations": [ /* DetectionExplanation[] — per active detector */ ]
    }
    /* …more frames… */
  ],

  "verdict": "true_positive",
  "user_rating": 4
}
```

### Top-level fields

| Field | Type | Notes |
|-------|------|-------|
| `timestamp` | RFC 3339 string (UTC) | When the recording was saved. |
| `trigger` | `"detection"` \| `"missed_event"` \| `"false_positive"` | Who decided to record. `detection` = detector fired. `missed_event` = user pressed F10 because the detector missed one. `false_positive` = user pressed F9. |
| `bfrb_type` | `"nail_biting"` \| `"nail_picking"` \| null | Null for `missed_event` recordings until labelled. |
| `confidence` | float (0..1) or null | Detector's max per-hand confidence at trigger time. |
| `explanation` | object or null | The trigger-frame `DetectionExplanation` — *which* signals contributed and which suppressions fired. Only present for `detection` triggers. |
| `frames` | array | One entry per saved frame (pre + trigger + post), ordered by `offset`. |
| `verdict` | `"true_positive"` \| `"false_positive"` \| `"unsure"` \| absent | **Added later** by the user via the alert notification, the history page, or the F9/F10 hotkeys. The cleanest training signal in this file. |
| `user_rating` | 1..5 or absent | Fuzzy quality slider, also user-supplied. Lower priority than `verdict`. |

### `frames[]` entries

| Field | Type | Notes |
|-------|------|-------|
| `offset` | int | Frame index relative to the trigger. Negative = before, 0 = trigger, positive = after. |
| `timestamp_ms` | int | ms since the detection thread started. |
| `hand_count` | int | Tracked hand count *after* the temporal tracker (not raw palm detections). |
| `hand_sides` | string[] | Per-hand `"left"` / `"right"` / `"unknown"`. |
| `hand_confidences` | float[] | One per hand, 0..1. |
| `face_detected` | bool | Whether the face mesh was usable this frame. |
| `pose_detected` | bool | Whether pose landmarks were usable this frame. |
| `overlay.hands[]` | `HandResult` | 21 hand landmarks per hand (MediaPipe convention), plus handedness + confidence. |
| `overlay.face` | `FaceResult` | 468 face-mesh landmarks. |
| `overlay.pose` | `PoseResult` | 33 BlazePose landmarks, each with `x, y, z, visibility, presence`. |
| `explanations` | `DetectionExplanation[]` | Per-detector breakdown of contributing signals and suppressions (`typing_suppression`, `chin_rest_suppression`, `eating_suppression`, …) — empty when no detection ran. |

Landmark coordinates are normalised to the frame size (`x`, `y` in `[0, 1]`).
Definitive landmark indices (fingertips, MCP joints, outer lip contour, etc.)
are listed in `src-tauri/src/detection/types.rs`.

## Reading a session

The recorder caps history at `history.max_events` (default 100, oldest pruned
on rotation), and old events are also pruned at startup.

Quick exploration:

```bash
HIST=~/.local/share/nailbite/history
ls "$HIST"                                       # event dirs
jq '. | {trigger, bfrb_type, verdict, confidence}' "$HIST"/*/event.json
jq '.frames[] | {offset, hand_count, face_detected}' "$HIST"/<event>/event.json
```

The trigger-frame raw JPEG is `frames/frame_+00.jpg`; the same frame with
landmark overlay rendered is `frames/frame_+00_annotated.jpg`.

## What's *not* in the data (yet)

These are open questions if you're thinking about training on this:

- **Inter-rater agreement.** Today there's a single user; `verdict` is a
  single subjective label per event. No second opinion, no held-out
  adjudicator.
- **Hard negatives.** Only events that *triggered* (or that the user
  explicitly recorded as missed) are saved. The ambient stream of "hand
  near face but not biting" — which is the hard part of this problem —
  has no clip representation.
- **Privacy-preserving export.** No tooling yet to export a labelled
  subset with faces blurred / dropped / replaced by landmarks-only. This
  is the obvious blocker for cross-user dataset sharing.
- **Pose-only events.** When the presence gate is closed (no face / no
  torso), the pipeline never reaches the behaviour detectors, so the
  history under-represents partial-frame poses.

## Schema stability

The format is JSON and additive: new keys can appear without breaking older
consumers. Anything that ever changes shape is called out in
[`CHANGELOG.md`](../CHANGELOG.md) (created on the first breaking change). The
on-disk frame format (JPEG) and the directory layout are stable from v0.1.0
onward.
