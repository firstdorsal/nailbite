//! Configuration commands.

use std::sync::Arc;

use tauri::State;
use tracing::info;

use crate::actions::sound::SoundAction;
use crate::actions::webhook::WebhookAction;
use crate::config::NailbiteConfig;
use crate::detection::behaviors::{nail_biting::NailBitingDetector, nail_picking::NailPickingDetector, BehaviorDetector};
use crate::detection::tracker::{BehaviorTracker, DetectionTracker};
use crate::errors::NailbiteError;
use crate::state::AppState;

/// Get the current configuration.
#[tauri::command]
pub fn get_config(state: State<'_, Arc<AppState>>) -> Result<NailbiteConfig, NailbiteError> {
    let config = state.config.read();
    Ok(config.clone())
}

/// Save configuration to file and update runtime state.
///
/// Implements config hot-reload (ARCH-3): rebuilds detectors, tracker,
/// and actions based on the new configuration without requiring app restart.
#[tauri::command]
pub fn save_config(
    config: NailbiteConfig,
    state: State<'_, Arc<AppState>>,
) -> Result<(), NailbiteError> {
    // Validate
    config.validate()?;

    // Save to file
    config.save("config.yaml")?;

    // Rebuild detectors based on new config
    let mut new_detectors: Vec<Box<dyn BehaviorDetector>> = Vec::new();

    if config.detection.behaviors.nail_biting.enabled {
        new_detectors.push(Box::new(NailBitingDetector::new(
            &config.detection.behaviors.nail_biting,
            config.detection.false_positive.chin_rest_suppression,
            config.detection.false_positive.typing_suppression,
        )));
        info!("Nail biting detector enabled (hot-reload)");
    }

    if config.detection.behaviors.nail_picking.enabled {
        new_detectors.push(Box::new(NailPickingDetector::new(
            &config.detection.behaviors.nail_picking,
            config.detection.false_positive.typing_suppression,
        )));
        info!("Nail picking detector enabled (hot-reload)");
    }

    // Rebuild tracker with new configuration
    let mut new_trackers = Vec::new();
    if config.detection.behaviors.nail_biting.enabled {
        new_trackers.push(BehaviorTracker::new(
            crate::detection::types::BfrbType::NailBiting,
            std::time::Duration::from_millis(config.detection.temporal.window_ms),
            config.detection.temporal.positive_ratio,
            config.detection.behaviors.nail_biting.confidence_threshold,
            std::time::Duration::from_secs(config.general.cooldown_seconds),
        ));
    }
    if config.detection.behaviors.nail_picking.enabled {
        new_trackers.push(BehaviorTracker::new(
            crate::detection::types::BfrbType::NailPicking,
            std::time::Duration::from_millis(config.detection.temporal.window_ms),
            config.detection.temporal.positive_ratio,
            config.detection.behaviors.nail_picking.confidence_threshold,
            std::time::Duration::from_secs(config.general.cooldown_seconds),
        ));
    }
    let new_tracker = DetectionTracker::new(new_trackers);

    // Update runtime state atomically
    {
        *state.detectors.write() = new_detectors;
        *state.tracker.write() = new_tracker;
    }

    // Update sound action
    {
        let new_sound = SoundAction::new(
            &config.actions.sound.file,
            config.actions.sound.volume,
            config.actions.sound.repeat,
        );
        *state.sound_action.lock() = new_sound;
    }

    // Update webhook action
    {
        let new_webhook = if config.actions.webhook.enabled && !config.actions.webhook.url.is_empty() {
            match WebhookAction::new(
                &config.actions.webhook.url,
                config.actions.webhook.timeout_ms,
                config.actions.webhook.headers.clone(),
            ) {
                Ok(action) => Some(action),
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to create webhook action, disabling");
                    None
                }
            }
        } else {
            None
        };
        *state.webhook_action.lock() = new_webhook;
    }

    // Update runtime config
    {
        *state.config.write() = config;
    }

    info!("Configuration saved and applied (hot-reload)");
    Ok(())
}
