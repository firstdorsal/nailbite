//! Label-driven analysis: turn user verdicts into threshold tuning suggestions.
//!
//! Walks the event history directory, reads each `event.json`, and produces
//! a confusion summary per BFRB plus a list of threshold suggestions. The
//! suggestions answer "if I had used threshold X, how would my labels split?"
//! using only the data already stored in events.
//!
//! Currently sweeps `confidence_threshold` (the peak-confidence cutoff inside
//! the temporal tracker). `proximity_threshold` sweep is a possible future
//! addition: the per-frame explanation has the data (`normalized_distance`,
//! `curl`) needed to recompute confidences for alternative proximity values.

use std::fs;
use std::sync::Arc;

use serde::Serialize;
use tauri::State;
use tracing::warn;

use crate::detection::types::{BfrbType, DetectionExplanation};
use crate::errors::NailbiteError;
use crate::paths::expand_tilde;
use crate::state::AppState;

use super::history::Verdict;

/// Per-event row used by the analyzer. One per labeled detection event.
#[derive(Debug, Clone)]
struct LabeledEvent {
    bfrb_type: String,
    confidence: f32,
    verdict: Verdict,
    /// Trigger-frame explanation, when present in `event.json`.
    /// Required for the proximity-threshold sweep; older events recorded
    /// before Track 1 may lack this field — they are silently excluded
    /// from the proximity sweep but still counted in the confidence sweep.
    explanation: Option<DetectionExplanation>,
}

#[derive(Debug, Clone, Copy, Serialize, Default)]
pub struct VerdictCounts {
    pub true_positive: usize,
    pub false_positive: usize,
    pub unsure: usize,
}

/// Single threshold suggestion: "what if we used `proposed_value`?"
#[derive(Debug, Clone, Serialize)]
pub struct ThresholdSuggestion {
    /// e.g. "confidence_threshold" — kept as a string so the frontend can
    /// label arbitrary parameters in future sweeps without backend changes.
    pub parameter: String,
    pub current_value: f32,
    pub proposed_value: f32,
    pub tp_kept: usize,
    pub tp_lost: usize,
    pub fp_kept: usize,
    pub fp_killed: usize,
    /// Precision over labeled (TP+FP) events that would still fire.
    pub precision: f32,
    /// Recall over labeled TPs.
    pub recall: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct BfrbAnalysis {
    pub bfrb_type: String,
    pub counts: VerdictCounts,
    /// Current `confidence_threshold` from `config.yaml`, included so the
    /// frontend can show the comparison row without a separate fetch.
    pub current_confidence_threshold: f32,
    /// Sorted ascending by `proposed_value`.
    pub confidence_suggestions: Vec<ThresholdSuggestion>,
    /// Current `proximity_threshold` from `config.yaml`.
    pub current_proximity_threshold: f32,
    /// Sorted ascending by `proposed_value`. Only populated for events
    /// that captured trigger-frame explanations.
    pub proximity_suggestions: Vec<ThresholdSuggestion>,
    /// Number of labeled events that lack explanation data and are
    /// therefore excluded from the proximity sweep.
    pub events_without_explanation: usize,
}

#[derive(Debug, Serialize)]
pub struct LabelAnalysis {
    pub total_labeled: usize,
    pub per_bfrb: Vec<BfrbAnalysis>,
}

/// Compute (precision, recall) for a hypothetical threshold value.
fn confusion_at(events: &[LabeledEvent], threshold: f32) -> ThresholdSuggestion {
    let mut tp_kept = 0usize;
    let mut tp_lost = 0usize;
    let mut fp_kept = 0usize;
    let mut fp_killed = 0usize;

    for e in events {
        let kept = e.confidence >= threshold;
        match e.verdict {
            Verdict::TruePositive => {
                if kept {
                    tp_kept += 1;
                } else {
                    tp_lost += 1;
                }
            }
            Verdict::FalsePositive => {
                if kept {
                    fp_kept += 1;
                } else {
                    fp_killed += 1;
                }
            }
            Verdict::Unsure => {} // Excluded from precision/recall.
        }
    }

    let kept_total = tp_kept + fp_kept;
    let precision = if kept_total == 0 {
        1.0
    } else {
        tp_kept as f32 / kept_total as f32
    };
    let total_tp = tp_kept + tp_lost;
    let recall = if total_tp == 0 {
        1.0
    } else {
        tp_kept as f32 / total_tp as f32
    };

    ThresholdSuggestion {
        parameter: "confidence_threshold".to_string(),
        current_value: 0.0, // filled in by caller
        proposed_value: threshold,
        tp_kept,
        tp_lost,
        fp_kept,
        fp_killed,
        precision,
        recall,
    }
}

fn config_confidence_threshold(state: &AppState, bfrb_type: BfrbType) -> f32 {
    let cfg = state.config.read();
    let b = &cfg.detection.behaviors;
    match bfrb_type {
        BfrbType::NailBiting => b.nail_biting.confidence_threshold,
        BfrbType::NailPicking => b.nail_picking.confidence_threshold,
        BfrbType::HairPulling => b.hair_pulling.confidence_threshold,
        BfrbType::SkinPicking => b.skin_picking.confidence_threshold,
        BfrbType::LipBiting => b.lip_biting.confidence_threshold,
    }
}

fn config_proximity_threshold(state: &AppState, bfrb_type: BfrbType) -> f32 {
    let cfg = state.config.read();
    let b = &cfg.detection.behaviors;
    match bfrb_type {
        BfrbType::NailBiting => b.nail_biting.proximity_threshold,
        BfrbType::NailPicking => b.nail_picking.proximity_threshold,
        BfrbType::HairPulling => b.hair_pulling.proximity_threshold,
        BfrbType::SkinPicking => b.skin_picking.proximity_threshold,
        BfrbType::LipBiting => b.lip_biting.proximity_threshold,
    }
}

/// Recompute the peak frame confidence under a hypothetical
/// `new_proximity` threshold, using the per-hand signals captured at
/// detection time. Mirrors the formulas in `behaviors/nail_biting.rs` and
/// `behaviors/nail_picking.rs`.
fn recompute_confidence_at_proximity(
    bfrb_type: BfrbType,
    explanation: &DetectionExplanation,
    new_proximity: f32,
) -> f32 {
    if new_proximity <= 0.0 {
        return 0.0;
    }
    // If a hard suppression fired (typing, chin rest), the score is 0
    // regardless of proximity.
    if !explanation.suppressions.is_empty() {
        return 0.0;
    }

    let mut max_conf = 0.0_f32;
    for sig in &explanation.hands {
        // Picking same-hand uses 0.5x the configured proximity threshold;
        // detect it as "no other hand_index has a paired (matching distance)
        // signal". Inter-hand picking emits two mirrored signals.
        let is_picking_same_hand = matches!(bfrb_type, BfrbType::NailPicking)
            && !explanation.hands.iter().any(|other| {
                !std::ptr::eq(other, sig)
                    && other.hand_index != sig.hand_index
                    && (other.normalized_distance - sig.normalized_distance).abs() < 1e-4
            });

        let effective_threshold = if is_picking_same_hand {
            new_proximity * 0.5
        } else {
            new_proximity
        };

        if sig.normalized_distance >= effective_threshold {
            continue;
        }
        let proximity_score = 1.0 - sig.normalized_distance / effective_threshold;
        let conf = match bfrb_type {
            BfrbType::NailBiting => {
                let curl = sig.curl.unwrap_or(0.0);
                proximity_score * (0.8 + 0.2 * curl)
            }
            BfrbType::NailPicking => (proximity_score + sig.bonus).min(1.0),
            _ => proximity_score,
        };
        if conf > max_conf {
            max_conf = conf;
        }
    }
    max_conf
}

/// Sweep proximity thresholds. For each candidate, recompute the peak
/// confidence per labeled event and apply the current `confidence_threshold`
/// to decide if the event would still fire.
fn sweep_proximity(
    events: &[LabeledEvent],
    bfrb: BfrbType,
    current_proximity: f32,
    confidence_threshold: f32,
) -> Vec<ThresholdSuggestion> {
    // Anchor the sweep at fixed values plus the current setting; use a
    // 0.05 grid spanning the typical operating range. The confidence
    // sweep used observed values, but here we don't have that natural
    // discretization, so a coarse grid + current is fine.
    let mut candidates: Vec<f32> = Vec::new();
    let mut v = 0.05_f32;
    while v <= 0.80 + 1e-6 {
        candidates.push(v);
        v += 0.05;
    }
    candidates.push(current_proximity);
    candidates.retain(|c| *c > 0.0 && *c <= 1.0);
    candidates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    candidates.dedup_by(|a, b| (*a - *b).abs() < 1e-3);

    candidates
        .into_iter()
        .map(|prox| {
            let mut tp_kept = 0usize;
            let mut tp_lost = 0usize;
            let mut fp_kept = 0usize;
            let mut fp_killed = 0usize;
            for e in events {
                let exp = match &e.explanation {
                    Some(e) => e,
                    None => continue,
                };
                let conf = recompute_confidence_at_proximity(bfrb, exp, prox);
                let kept = conf >= confidence_threshold;
                match e.verdict {
                    Verdict::TruePositive => {
                        if kept { tp_kept += 1 } else { tp_lost += 1 }
                    }
                    Verdict::FalsePositive => {
                        if kept { fp_kept += 1 } else { fp_killed += 1 }
                    }
                    Verdict::Unsure => {}
                }
            }
            let kept_total = tp_kept + fp_kept;
            let precision = if kept_total == 0 {
                1.0
            } else {
                tp_kept as f32 / kept_total as f32
            };
            let total_tp = tp_kept + tp_lost;
            let recall = if total_tp == 0 {
                1.0
            } else {
                tp_kept as f32 / total_tp as f32
            };
            ThresholdSuggestion {
                parameter: "proximity_threshold".to_string(),
                current_value: current_proximity,
                proposed_value: prox,
                tp_kept,
                tp_lost,
                fp_kept,
                fp_killed,
                precision,
                recall,
            }
        })
        .collect()
}

fn parse_bfrb(s: &str) -> Option<BfrbType> {
    match s {
        "nail_biting" => Some(BfrbType::NailBiting),
        "nail_picking" => Some(BfrbType::NailPicking),
        "hair_pulling" => Some(BfrbType::HairPulling),
        "skin_picking" => Some(BfrbType::SkinPicking),
        "lip_biting" => Some(BfrbType::LipBiting),
        _ => None,
    }
}

/// Read the labeled events from the history directory.
fn load_labeled_events(history_dir: &std::path::Path) -> Vec<LabeledEvent> {
    let Ok(entries) = fs::read_dir(history_dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let json_path = path.join("event.json");
        let Ok(json) = fs::read_to_string(&json_path) else {
            continue;
        };
        let Ok(meta) = serde_json::from_str::<serde_json::Value>(&json) else {
            continue;
        };

        let trigger = meta
            .get("trigger")
            .and_then(|t| t.as_str())
            .unwrap_or("");
        if trigger != "detection" {
            continue;
        }

        let bfrb = meta.get("bfrb_type").and_then(|t| t.as_str());
        #[allow(clippy::cast_possible_truncation)]
        let conf = meta
            .get("confidence")
            .and_then(|c| c.as_f64())
            .map(|c| c as f32);

        // Verdict can be in the new field, or the legacy user_label format.
        let verdict = meta
            .get("verdict")
            .and_then(|v| serde_json::from_value::<Verdict>(v.clone()).ok())
            .or_else(|| {
                meta.get("user_label").and_then(|v| v.as_str()).and_then(
                    |label| match label {
                        "correct" => Some(Verdict::TruePositive),
                        "incorrect" => Some(Verdict::FalsePositive),
                        _ => None,
                    },
                )
            });

        if let (Some(b), Some(c), Some(v)) = (bfrb, conf, verdict) {
            let explanation = meta
                .get("explanation")
                .and_then(|e| serde_json::from_value::<DetectionExplanation>(e.clone()).ok());
            out.push(LabeledEvent {
                bfrb_type: b.to_string(),
                confidence: c,
                verdict: v,
                explanation,
            });
        }
    }
    out
}

/// Sweep candidate confidence thresholds and return Pareto-helpful suggestions.
fn sweep_confidence(
    events: &[LabeledEvent],
    current: f32,
) -> Vec<ThresholdSuggestion> {
    // Use the actual labeled-event confidences as candidates plus a few fixed
    // anchors. Sweeping observed values guarantees we hit every transition.
    let mut candidates: Vec<f32> = events.iter().map(|e| e.confidence).collect();
    candidates.extend([0.30_f32, 0.40, 0.50, 0.60, 0.70, 0.80, current]);
    candidates.retain(|v| (0.0..=1.0).contains(v));
    // Dedup with epsilon.
    candidates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    candidates.dedup_by(|a, b| (*a - *b).abs() < 1e-3);

    candidates
        .into_iter()
        .map(|t| {
            let mut s = confusion_at(events, t);
            s.current_value = current;
            s
        })
        .collect()
}

/// Walk the saved event history and return per-BFRB confusion + threshold
/// sweeps. Events without a verdict are counted but excluded from sweeps.
#[tauri::command]
pub fn analyze_labels(
    state: State<'_, Arc<AppState>>,
) -> Result<LabelAnalysis, NailbiteError> {
    let cfg = state.config.read();
    let history_dir = expand_tilde(&cfg.history.dir);
    drop(cfg);

    if !history_dir.exists() {
        return Ok(LabelAnalysis {
            total_labeled: 0,
            per_bfrb: Vec::new(),
        });
    }

    let events = load_labeled_events(&history_dir);

    // Group by bfrb_type.
    let mut by_bfrb: std::collections::BTreeMap<String, Vec<LabeledEvent>> =
        std::collections::BTreeMap::new();
    for e in events.iter().cloned() {
        by_bfrb.entry(e.bfrb_type.clone()).or_default().push(e);
    }

    let mut per_bfrb = Vec::new();
    for (bfrb, evs) in by_bfrb {
        let mut counts = VerdictCounts::default();
        for e in &evs {
            match e.verdict {
                Verdict::TruePositive => counts.true_positive += 1,
                Verdict::FalsePositive => counts.false_positive += 1,
                Verdict::Unsure => counts.unsure += 1,
            }
        }

        let parsed_bfrb = parse_bfrb(&bfrb);
        let current_conf = parsed_bfrb
            .map(|b| config_confidence_threshold(&state, b))
            .unwrap_or_else(|| {
                warn!(bfrb = %bfrb, "unknown bfrb in label analysis");
                0.3
            });
        let current_prox = parsed_bfrb
            .map(|b| config_proximity_threshold(&state, b))
            .unwrap_or(0.35);

        let confidence_sweep = sweep_confidence(&evs, current_conf);
        let proximity_sweep = parsed_bfrb
            .map(|b| sweep_proximity(&evs, b, current_prox, current_conf))
            .unwrap_or_default();

        let events_without_explanation =
            evs.iter().filter(|e| e.explanation.is_none()).count();

        per_bfrb.push(BfrbAnalysis {
            bfrb_type: bfrb,
            counts,
            current_confidence_threshold: current_conf,
            confidence_suggestions: confidence_sweep,
            current_proximity_threshold: current_prox,
            proximity_suggestions: proximity_sweep,
            events_without_explanation,
        });
    }

    Ok(LabelAnalysis {
        total_labeled: events.len(),
        per_bfrb,
    })
}

/// Result of a dataset export.
#[derive(Debug, Serialize)]
pub struct DatasetExportResult {
    pub target_dir: String,
    pub events_exported: usize,
    pub frames_copied: usize,
    pub manifest_path: String,
}

/// Copy labeled events to `target_dir` as a flat dataset:
///
/// ```text
/// target_dir/
///   dataset.jsonl       # one line per event
///   <event_id>/
///     event.json
///     frames/...        # raw + annotated frames
/// ```
///
/// Each line in `dataset.jsonl` includes the verdict, confidence, BFRB type,
/// trigger explanation, and a relative path to the frames directory — enough
/// for an offline classifier to consume without re-reading event.json.
///
/// Only events with a non-`Unsure` verdict are exported (Unsure adds noise
/// to a training set).
#[tauri::command]
pub fn export_labeled_dataset(
    state: State<'_, Arc<AppState>>,
    target_dir: String,
) -> Result<DatasetExportResult, NailbiteError> {
    let cfg = state.config.read();
    let history_dir = expand_tilde(&cfg.history.dir);
    drop(cfg);

    let target = expand_tilde(&std::path::PathBuf::from(&target_dir));
    fs::create_dir_all(&target)?;

    let manifest_path = target.join("dataset.jsonl");
    let mut manifest = String::new();

    let mut events_exported = 0usize;
    let mut frames_copied = 0usize;

    if !history_dir.exists() {
        fs::write(&manifest_path, &manifest)?;
        return Ok(DatasetExportResult {
            target_dir: target.display().to_string(),
            events_exported: 0,
            frames_copied: 0,
            manifest_path: manifest_path.display().to_string(),
        });
    }

    for entry in fs::read_dir(&history_dir)?.flatten() {
        let src = entry.path();
        if !src.is_dir() {
            continue;
        }
        let dir_name = match src.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };

        let json_path = src.join("event.json");
        let Ok(json) = fs::read_to_string(&json_path) else {
            continue;
        };
        let Ok(meta) = serde_json::from_str::<serde_json::Value>(&json) else {
            continue;
        };

        let verdict = meta
            .get("verdict")
            .and_then(|v| serde_json::from_value::<Verdict>(v.clone()).ok());
        let verdict = match verdict {
            Some(Verdict::TruePositive) | Some(Verdict::FalsePositive) => verdict.unwrap(),
            _ => continue, // Skip Unsure / unlabeled.
        };

        // Copy the event directory (event.json + frames/) into target.
        let dst = target.join(&dir_name);
        fs::create_dir_all(&dst)?;
        fs::copy(&json_path, dst.join("event.json"))?;

        let src_frames = src.join("frames");
        let dst_frames = dst.join("frames");
        if src_frames.exists() {
            fs::create_dir_all(&dst_frames)?;
            for f in fs::read_dir(&src_frames)?.flatten() {
                let p = f.path();
                if p.is_file() {
                    if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                        fs::copy(&p, dst_frames.join(name))?;
                        frames_copied += 1;
                    }
                }
            }
        }

        // One JSONL line per event with the bits a trainer needs.
        let line = serde_json::json!({
            "event_id": dir_name,
            "verdict": match verdict {
                Verdict::TruePositive => "true_positive",
                Verdict::FalsePositive => "false_positive",
                Verdict::Unsure => "unsure",
            },
            "bfrb_type": meta.get("bfrb_type").cloned(),
            "confidence": meta.get("confidence").cloned(),
            "explanation": meta.get("explanation").cloned(),
            "frames_dir": format!("{dir_name}/frames"),
            "verdict_reason": meta.get("verdict_reason").cloned(),
        });
        manifest.push_str(&serde_json::to_string(&line).map_err(|e| {
            NailbiteError::Camera(format!("Failed to serialize manifest line: {e}"))
        })?);
        manifest.push('\n');
        events_exported += 1;
    }

    fs::write(&manifest_path, &manifest)?;

    Ok(DatasetExportResult {
        target_dir: target.display().to_string(),
        events_exported,
        frames_copied,
        manifest_path: manifest_path.display().to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(bfrb: &str, conf: f32, v: Verdict) -> LabeledEvent {
        LabeledEvent {
            bfrb_type: bfrb.into(),
            confidence: conf,
            verdict: v,
            explanation: None,
        }
    }

    fn ev_with_exp(
        bfrb: &str,
        conf: f32,
        v: Verdict,
        exp: DetectionExplanation,
    ) -> LabeledEvent {
        LabeledEvent {
            bfrb_type: bfrb.into(),
            confidence: conf,
            verdict: v,
            explanation: Some(exp),
        }
    }

    #[test]
    fn confusion_separates_above_below_threshold() {
        let evs = vec![
            ev("nail_biting", 0.9, Verdict::TruePositive),
            ev("nail_biting", 0.5, Verdict::TruePositive),
            ev("nail_biting", 0.4, Verdict::FalsePositive),
            ev("nail_biting", 0.95, Verdict::FalsePositive),
        ];
        let s = confusion_at(&evs, 0.6);
        assert_eq!(s.tp_kept, 1); // 0.9
        assert_eq!(s.tp_lost, 1); // 0.5
        assert_eq!(s.fp_killed, 1); // 0.4
        assert_eq!(s.fp_kept, 1); // 0.95
        // precision = TP/(TP+FP) = 1/2 = 0.5
        assert!((s.precision - 0.5).abs() < 1e-6);
        // recall = TP_kept / total_TP = 1/2
        assert!((s.recall - 0.5).abs() < 1e-6);
    }

    #[test]
    fn confusion_excludes_unsure() {
        let evs = vec![
            ev("x", 0.9, Verdict::TruePositive),
            ev("x", 0.9, Verdict::Unsure),
            ev("x", 0.1, Verdict::Unsure),
        ];
        let s = confusion_at(&evs, 0.5);
        assert_eq!(s.tp_kept, 1);
        assert_eq!(s.fp_kept, 0);
        assert_eq!(s.fp_killed, 0);
    }

    #[test]
    fn load_labeled_events_picks_up_legacy_user_label() {
        // event.json has the old format (user_label string). Loader should
        // surface it as the right verdict.
        let dir = tempfile::tempdir().unwrap();
        let event_dir = dir.path().join("20260101_001");
        std::fs::create_dir_all(&event_dir).unwrap();

        let meta = serde_json::json!({
            "trigger": "detection",
            "bfrb_type": "nail_biting",
            "confidence": 0.7,
            "user_label": "incorrect"
        });
        std::fs::write(
            event_dir.join("event.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();

        let evs = load_labeled_events(dir.path());
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].verdict, Verdict::FalsePositive);
        assert!((evs[0].confidence - 0.7).abs() < 1e-6);
    }

    #[test]
    fn proximity_sweep_kills_fp_when_threshold_tightened() {
        // Two events: a TP with very close fingertip and an FP with the
        // fingertip near the proximity boundary. Tightening proximity
        // should keep the TP and kill the FP without losing recall.
        use crate::detection::types::HandSignal;

        let exp_tp = DetectionExplanation {
            bfrb_type: BfrbType::NailBiting,
            hands: vec![HandSignal {
                hand_index: 0,
                side: None,
                normalized_distance: 0.05,
                distance_threshold: 0.35,
                contributing_fingertip: Some(8),
                partner_fingertip: None,
                curl: Some(0.7),
                bonus: 0.0,
                confidence: 0.85,
            }],
            suppressions: vec![],
            frame_confidence: 0.85,
        };
        let exp_fp = DetectionExplanation {
            bfrb_type: BfrbType::NailBiting,
            hands: vec![HandSignal {
                hand_index: 0,
                side: None,
                normalized_distance: 0.30,
                distance_threshold: 0.35,
                contributing_fingertip: Some(8),
                partner_fingertip: None,
                curl: Some(0.4),
                bonus: 0.0,
                confidence: 0.13,
            }],
            suppressions: vec![],
            frame_confidence: 0.13,
        };

        let evs = vec![
            ev_with_exp("nail_biting", 0.85, Verdict::TruePositive, exp_tp),
            ev_with_exp("nail_biting", 0.13, Verdict::FalsePositive, exp_fp),
        ];

        // confidence_threshold=0.3, sweep proximity from 0.35 down.
        let sweep = sweep_proximity(&evs, BfrbType::NailBiting, 0.35, 0.30);

        // At very tight proximity the FP should be killed (its 0.30
        // normalized distance falls outside any threshold <= 0.30) while
        // the TP at 0.05 still produces high confidence.
        let row_at_25 = sweep
            .iter()
            .find(|s| (s.proposed_value - 0.25).abs() < 1e-3)
            .expect("proximity 0.25 candidate present");
        assert_eq!(row_at_25.fp_killed, 1, "tightening should kill the FP");
        assert_eq!(row_at_25.tp_kept, 1, "TP should still fire");
    }

    #[test]
    fn load_labeled_events_skips_unlabeled() {
        let dir = tempfile::tempdir().unwrap();
        let event_dir = dir.path().join("20260101_002");
        std::fs::create_dir_all(&event_dir).unwrap();

        let meta = serde_json::json!({
            "trigger": "detection",
            "bfrb_type": "nail_biting",
            "confidence": 0.7,
            // no verdict / user_label
        });
        std::fs::write(
            event_dir.join("event.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();

        let evs = load_labeled_events(dir.path());
        assert!(evs.is_empty(), "unlabeled events should be skipped");
    }

    #[test]
    fn sweep_includes_observed_values() {
        let evs = vec![
            ev("x", 0.55, Verdict::TruePositive),
            ev("x", 0.62, Verdict::FalsePositive),
        ];
        let sweep = sweep_confidence(&evs, 0.5);
        // Both observed values should be in there (deduped).
        let has_observed = |v: f32| {
            sweep
                .iter()
                .any(|s| (s.proposed_value - v).abs() < 1e-3)
        };
        assert!(has_observed(0.55));
        assert!(has_observed(0.62));
    }
}
