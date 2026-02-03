//! HTTP webhook action using reqwest.
//!
//! Sends a POST request with JSON payload when a BFRB is detected.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use tracing::{debug, warn};

use crate::actions::types::Action;
use crate::detection::types::DetectionEvent;
use crate::errors::ActionError;

#[derive(Serialize)]
struct WebhookPayload {
    event: String,
    bfrb_type: String,
    confidence: f32,
    camera_id: String,
    duration_ms: u64,
}

pub struct WebhookAction {
    url: String,
    headers: HashMap<String, String>,
    active: Arc<AtomicBool>,
    client: reqwest::blocking::Client,
}

impl WebhookAction {
    pub fn new(
        url: &str,
        timeout_ms: u64,
        headers: HashMap<String, String>,
    ) -> Result<Self, ActionError> {
        // Validate URL scheme to prevent SSRF against non-HTTP services.
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Err(ActionError::Webhook(format!(
                "Webhook URL must use http:// or https:// scheme, got: {url}"
            )));
        }

        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_millis(timeout_ms))
            .build()
            .map_err(|e| ActionError::Webhook(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self {
            url: url.to_string(),
            headers,
            active: Arc::new(AtomicBool::new(false)),
            client,
        })
    }
}

impl Action for WebhookAction {
    fn start(&mut self, event: &DetectionEvent) -> Result<(), ActionError> {
        if self.active.load(Ordering::Relaxed) {
            return Ok(());
        }

        debug!(url = %self.url, bfrb_type = %event.bfrb_type, "Sending webhook");

        let payload = WebhookPayload {
            event: "bfrb_detected".to_string(),
            bfrb_type: event.bfrb_type.to_string(),
            confidence: event.confidence,
            camera_id: event.camera_id.clone(),
            // Duration will never exceed u64::MAX milliseconds in practice.
            #[allow(clippy::cast_possible_truncation)]
            duration_ms: event.duration.as_millis() as u64,
        };

        let mut request = self.client.post(&self.url).json(&payload);

        for (key, value) in &self.headers {
            request = request.header(key, value);
        }

        match request.send() {
            Ok(response) => {
                if !response.status().is_success() {
                    warn!(
                        status = %response.status(),
                        "Webhook returned non-success status"
                    );
                }
            }
            Err(e) => {
                warn!(error = %e, "Webhook request failed");
                return Err(ActionError::Webhook(e.to_string()));
            }
        }

        self.active.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn stop(&mut self) -> Result<(), ActionError> {
        if !self.active.load(Ordering::Relaxed) {
            return Ok(());
        }

        // Optionally send a "stopped" webhook.
        debug!("Webhook action stopped");
        self.active.store(false, Ordering::Relaxed);
        Ok(())
    }

    fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn webhook_action_starts_inactive() {
        let action =
            WebhookAction::new("http://localhost:9999/test", 5000, HashMap::new()).unwrap();
        assert!(!action.is_active());
    }

    #[test]
    fn webhook_rejects_non_http_url() {
        let result = WebhookAction::new("file:///etc/passwd", 5000, HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn webhook_rejects_ftp_url() {
        let result = WebhookAction::new("ftp://evil.com/data", 5000, HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn webhook_accepts_https_url() {
        let result = WebhookAction::new("https://hooks.example.com/bfrb", 5000, HashMap::new());
        assert!(result.is_ok());
    }

    #[test]
    fn webhook_payload_serialization() {
        let payload = WebhookPayload {
            event: "bfrb_detected".to_string(),
            bfrb_type: "Nail Biting".to_string(),
            confidence: 0.85,
            camera_id: "main".to_string(),
            duration_ms: 2500,
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("bfrb_detected"));
        assert!(json.contains("Nail Biting"));
    }
}
