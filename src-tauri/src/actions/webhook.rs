//! HTTP webhook action using reqwest.
//!
//! Sends a POST request with JSON payload when a BFRB is detected.

use std::collections::HashMap;
use std::net::{IpAddr, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use tracing::{debug, warn};
use url::Url;

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

/// Check if an IP address is internal/private (SECURITY-2: SSRF protection).
fn is_internal_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_broadcast()
                || v4.is_documentation()
                || v4.is_unspecified()
                // AWS metadata endpoint
                || v4.octets() == [169, 254, 169, 254]
                // Class E reserved
                || v4.octets()[0] >= 240
        }
        IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unspecified()
                // IPv6 link-local
                || (v6.segments()[0] & 0xffc0) == 0xfe80
                // IPv6 unique local
                || (v6.segments()[0] & 0xfe00) == 0xfc00
        }
    }
}

/// Validate that URL doesn't point to internal network (SECURITY-2).
fn validate_url_not_internal(url: &str) -> Result<(), ActionError> {
    let parsed = Url::parse(url)
        .map_err(|e| ActionError::Webhook(format!("Invalid URL: {e}")))?;

    let host = parsed.host_str()
        .ok_or_else(|| ActionError::Webhook("URL has no host".to_string()))?;

    // Check if host is an IP address directly
    if let Ok(ip) = host.parse::<IpAddr>() {
        if is_internal_ip(ip) {
            return Err(ActionError::Webhook(format!(
                "Webhook URL points to internal IP address: {host}"
            )));
        }
        return Ok(());
    }

    // Resolve hostname and check all resulting IPs
    let port = parsed.port().unwrap_or(if parsed.scheme() == "https" { 443 } else { 80 });
    let addr = format!("{host}:{port}");

    if let Ok(addrs) = addr.to_socket_addrs() {
        for socket_addr in addrs {
            if is_internal_ip(socket_addr.ip()) {
                return Err(ActionError::Webhook(format!(
                    "Webhook URL resolves to internal IP address: {host} -> {}",
                    socket_addr.ip()
                )));
            }
        }
    }
    // If we can't resolve, allow it (DNS might fail temporarily)

    Ok(())
}

/// Validate header values for CRLF injection (SECURITY-3).
fn validate_header_value(key: &str, value: &str) -> Result<(), ActionError> {
    if value.contains('\r') || value.contains('\n') {
        return Err(ActionError::Webhook(format!(
            "Header value for '{key}' contains CRLF characters (potential header injection)"
        )));
    }
    Ok(())
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

        // Validate URL doesn't point to internal networks (SECURITY-2)
        validate_url_not_internal(url)?;

        // Validate header values for CRLF injection (SECURITY-3)
        for (key, value) in &headers {
            validate_header_value(key, value)?;
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
        // Use a valid external URL (example.com is reserved for documentation)
        let action =
            WebhookAction::new("https://hooks.example.com/test", 5000, HashMap::new()).unwrap();
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
    fn webhook_rejects_localhost() {
        let result = WebhookAction::new("http://localhost:9999/test", 5000, HashMap::new());
        match result {
            Ok(_) => panic!("Expected error for localhost URL"),
            Err(e) => {
                let msg = e.to_string();
                assert!(msg.contains("internal IP") || msg.contains("localhost"));
            }
        }
    }

    #[test]
    fn webhook_rejects_private_ip() {
        let result = WebhookAction::new("http://192.168.1.1:8080/hook", 5000, HashMap::new());
        match result {
            Ok(_) => panic!("Expected error for private IP"),
            Err(e) => assert!(e.to_string().contains("internal IP")),
        }
    }

    #[test]
    fn webhook_rejects_aws_metadata() {
        let result = WebhookAction::new("http://169.254.169.254/latest/meta-data", 5000, HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn webhook_rejects_crlf_in_headers() {
        let mut headers = HashMap::new();
        headers.insert("X-Custom".to_string(), "value\r\nX-Injected: evil".to_string());
        let result = WebhookAction::new("https://hooks.example.com/bfrb", 5000, headers);
        match result {
            Ok(_) => panic!("Expected error for CRLF in headers"),
            Err(e) => assert!(e.to_string().contains("CRLF")),
        }
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

    #[test]
    fn is_internal_ip_detects_loopback() {
        assert!(is_internal_ip("127.0.0.1".parse().unwrap()));
        assert!(is_internal_ip("::1".parse().unwrap()));
    }

    #[test]
    fn is_internal_ip_detects_private() {
        assert!(is_internal_ip("10.0.0.1".parse().unwrap()));
        assert!(is_internal_ip("172.16.0.1".parse().unwrap()));
        assert!(is_internal_ip("192.168.1.1".parse().unwrap()));
    }

    #[test]
    fn is_internal_ip_allows_public() {
        // 8.8.8.8 is Google's public DNS
        assert!(!is_internal_ip("8.8.8.8".parse().unwrap()));
    }
}
