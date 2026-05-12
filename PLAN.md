# Plan: Detection Transparency, Labeling, and Label-Driven Tuning ✅

Three coordinated tracks that extend the existing event-history infrastructure
into a full feedback loop: explain detections, label them, then use the labels.

## Track 1 — Signal Transparency ✅

Surface the contributing signals (proximity, curl, suppressions, etc.) at the
moment of detection and during review, instead of only `(bfrb_type, confidence)`.

- ✅ T1.1 `DetectionExplanation`, `HandSignal`, `SuppressionReason` in
  `detection/types.rs`.
- ✅ T1.2 `BehaviorDetector::analyze_frame_explained`. `analyze_frame` defers
  to it. Both detectors populate.
- ✅ T1.3 `BehaviorTracker.peak_explanation` carries through to `DetectionEvent`.
- ✅ T1.4 Per-frame + trigger-frame explanations in `event.json`. `commands/history.rs`
  exposes them on `EventHistoryDetail`.
- ✅ T1.5 `FrameUpdateEvent.current_signals`, `DetectionEventResult.explanation`,
  `bfrb-detected` payload extended.
- ✅ T1.6 `<SignalsPanel>` live on `PreviewPage`, fed by `currentSignals`.
- ✅ T1.7 `<AlertModal>` renders the trigger explanation inline.
- ✅ T1.8 `<SignalTimeline>` SVG chart + per-frame `<SignalsPanel>` in the
  detail dialog.
- ✅ T1.9 +7 tests: detector explanation shape (biting + picking + typing
  suppression), tracker peak preservation + clear, event.json round-trip.

## Track 2 — Explicit TP/FP Labeling ✅

Replace ambiguous 1-5 rating with explicit verdicts (kept alongside as severity).
Make labels reachable in the moment, not only via History.

- ✅ T2.1 `Verdict` enum + `verdict_reason` persisted in `event.json`.
  `EventHistorySummary.verdict`, `EventHistoryDetail.verdict/verdict_reason`.
- ✅ T2.2 `set_event_verdict` Tauri command. `rate_event` retained.
- ✅ T2.3 `<AlertModal>` "Was this right?" 3-button row. Resolves the
  `event_id` via `find_recent_event_for_alert` and persists verdict.
- ✅ T2.4 `find_recent_event_for_alert(bfrb_type)` — newest matching detection.
- ✅ T2.5 `<VerdictPills>` with reason `<Input>` in the detail dialog
  (above the rating slider). Reason persists on blur / Enter.
- ✅ T2.6 Verdict counts (TP / FP / Unsure / Unlabeled) badge row above list.
- ✅ T2.7 +4 tests in `commands/history`: legacy `user_label` migration,
  new-format read, prefer-new-over-legacy, round-trip preservation.

## Track 3 — Use the Labels ✅

Turn labels into action: first as threshold tuning suggestions, then as a clean
exportable dataset.

- ✅ T3.1 `analyze_labels` (`commands/labels.rs`) — per-BFRB confusion counts
  + sorted `confidence_threshold` sweep using the labeled events' own
  confidence values as candidates.
- ✅ T3.2 Threshold suggestion: `confusion_at` + `sweep_confidence` produce
  precision/recall + (TP kept, TP lost, FP killed, FP kept). The Insights
  page picks Pareto-helpful highlights (best precision @ recall ≥ 0.5,
  best recall, best F1).
- ✅ T3.3 `<InsightsPage>` route + sidebar item. One-click "Apply" patches
  the right `confidence_threshold` in `NailbiteConfig` and calls `save_config`
  for hot-reload, then re-runs the analysis.
- ✅ T3.4 `export_labeled_dataset(target_dir)` — copies labeled (TP/FP)
  event dirs + writes `dataset.jsonl` with verdict, confidence, explanation,
  and frames-dir pointer.
- ✅ T3.5 +5 tests in `commands/labels`: confusion threshold split, unsure
  excluded, sweep includes observed values, legacy `user_label` migration,
  unlabeled events skipped.

## Follow-ups landed after the main 21-task scope ✅

- ✅ **Live signal overlay** on `<LandmarkCanvas>`: draws a red line from
  the contributing fingertip to its target (mouth for biting, partner
  fingertip for picking) with a `ratio×` distance label. Dashed/grey when
  below threshold, solid/red when firing. Inter-hand pair deduplication
  built in. Wired through `currentSignals` from `useCamera`.
- ✅ **Hotkeys on `<AlertModal>` verdict buttons**: `1` = TP, `2` = FP,
  `3` = unsure (in addition to existing `Enter` / `Escape`). Inputs
  guarded so typing in fields doesn't trigger them. Visual `kbd` chip
  on each button.
- ✅ **`proximity_threshold` sweep** in `analyze_labels` — recomputes
  per-event confidence from the stored `normalized_distance` + `curl` +
  `bonus`, applying the same formulas as the live detectors. Handles
  same-hand picking's 0.5× threshold scaling. Inheritance fields:
  `current_proximity_threshold`, `proximity_suggestions`,
  `events_without_explanation` (older events without captured
  explanations are excluded with a UI note).
- ✅ Insights page now has separate "Confidence threshold" and
  "Proximity threshold" sections with independent Apply buttons that
  patch the right field in `NailbiteConfig` and `save_config`.
- ✅ +1 test: `proximity_sweep_kills_fp_when_threshold_tightened`.

## Verification

- 159 backend tests pass.
- 13 frontend tests pass.
- `cargo clippy --lib` clean.
- `pnpm build` produces 452 kB JS bundle.
- `cargo build --lib` succeeds.

## Sequencing & Why

Track 1 first: it makes the rest meaningful. Without explanation data,
labels are blind ("yes/no" with no feature vector), and threshold suggestions
in Track 3 can't reason about anything except confidence.

Track 2 second: rich labels feed Track 3's analyzer.

Track 3 last, in two halves: 3.1–3.3 deliver value without ML; 3.4 is the
opt-in dataset export for someone who later wants to train a small classifier.

## Notes

- No new heavy dependencies. Plotting in T1.8 / T3.3 with existing recharts
  if already present, otherwise a flat SVG.
- Keep `analyze_frame` as the canonical path for the trait. The explained
  variant is additive, default-implemented, opt-in.
- Per-frame explanations must stay cheap — they reuse values already computed
  inside the detector; we just stop discarding them.
