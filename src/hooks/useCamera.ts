import { useState, useEffect, useRef, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import type { HandDetection, FaceDetection, PoseDetection } from "@/types";

interface FrameUpdateEvent {
  camera_id: string;
  role: string;
  frame_base64: string;
  width: number;
  height: number;
  hands: HandDetection[];
  face: FaceDetection | null;
  pose: PoseDetection | null;
  detections: Array<{
    bfrb_type: string;
    confidence: number;
    timestamp: string;
  }>;
  alert_active: boolean;
  paused: boolean;
  timestamp_ms: number;
}

/** State for a single camera. */
export interface CameraState {
  cameraId: string;
  role: "primary" | "auxiliary";
  frameBase64: string;
  width: number;
  height: number;
  hands: HandDetection[];
  face: FaceDetection | null;
  pose: PoseDetection | null;
  timestampMs: number;
}

interface UseCameraOptions {
  enabled?: boolean;
  /** Filter to only receive updates for specific camera IDs */
  cameraFilter?: string[];
}

interface UseCameraResult {
  /** Map of camera ID to camera state */
  cameras: Map<string, CameraState>;
  /** Primary camera state (convenience accessor) */
  primaryCamera: CameraState | null;
  /** Whether an alert is currently active (from any camera) */
  alertActive: boolean;
  /** Whether detection is paused */
  paused: boolean;
  /** Error message if any */
  error: string | null;
  /** Whether cameras are streaming */
  isStreaming: boolean;
  /** Start all cameras */
  startCamera: () => Promise<void>;
  /** Stop all cameras */
  stopCamera: () => Promise<void>;
  /** Get ref for a specific camera's image element */
  getImageRef: (cameraId: string) => React.RefObject<HTMLImageElement | null>;
}

/**
 * Hook for managing camera streaming and detection state.
 *
 * Connects to the Rust backend via Tauri events to receive frame updates,
 * hand/face/pose detections, and alert states from all configured cameras.
 *
 * @param options - Configuration options
 * @param options.enabled - Whether to start camera automatically (default: true)
 * @param options.cameraFilter - Filter to only receive updates for specific camera IDs
 * @returns Camera state and control functions
 *
 * @example
 * ```tsx
 * const { primaryCamera, alertActive, startCamera, stopCamera } = useCamera();
 * ```
 */
export function useCamera(options: UseCameraOptions = {}): UseCameraResult {
  const { enabled = true, cameraFilter } = options;

  const imageRefs = useRef<Map<string, React.RefObject<HTMLImageElement | null>>>(new Map());
  const unlistenRef = useRef<UnlistenFn | null>(null);
  // Track camera filter in ref to avoid recreating callbacks (TECH-6)
  const cameraFilterRef = useRef(cameraFilter);
  cameraFilterRef.current = cameraFilter;

  const [cameras, setCameras] = useState<Map<string, CameraState>>(new Map());
  const [alertActive, setAlertActive] = useState(false);
  const [paused, setPaused] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);

  const getImageRef = useCallback((cameraId: string) => {
    if (!imageRefs.current.has(cameraId)) {
      imageRefs.current.set(cameraId, { current: null });
    }
    return imageRefs.current.get(cameraId)!;
  }, []);

  // Stable callback using ref for filter (TECH-6)
  const startCamera = useCallback(async () => {
    try {
      setError(null);

      // Set up event listener for frame updates
      const unlisten = await listen<FrameUpdateEvent>("frame-update", (event) => {
        const data = event.payload;

        // Apply camera filter if specified (read from ref for stability)
        const filter = cameraFilterRef.current;
        if (filter && !filter.includes(data.camera_id)) {
          return;
        }

        // Update image source with base64 frame
        const ref = imageRefs.current.get(data.camera_id);
        if (ref?.current && data.frame_base64) {
          ref.current.src = `data:image/jpeg;base64,${data.frame_base64}`;
        }

        // Update camera state
        setCameras((prev) => {
          const next = new Map(prev);
          next.set(data.camera_id, {
            cameraId: data.camera_id,
            role: data.role as "primary" | "auxiliary",
            frameBase64: data.frame_base64,
            width: data.width,
            height: data.height,
            hands: data.hands,
            face: data.face,
            pose: data.pose,
            timestampMs: data.timestamp_ms,
          });
          return next;
        });

        // Update global state
        setAlertActive(data.alert_active);
        setPaused(data.paused);
      });

      unlistenRef.current = unlisten;

      // Start the Rust camera capture
      await invoke("start_camera");

      setIsStreaming(true);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      setError(message);
      console.error("Camera error:", e);
    }
  }, []); // No dependencies - uses refs for dynamic values

  const stopCamera = useCallback(async () => {
    try {
      // Stop listening for events
      if (unlistenRef.current) {
        unlistenRef.current();
        unlistenRef.current = null;
      }

      // Stop the Rust camera capture
      await invoke("stop_camera");

      // Clear all image refs
      imageRefs.current.forEach((ref) => {
        if (ref.current) {
          ref.current.src = "";
        }
      });

      setIsStreaming(false);
      setCameras(new Map());
      setAlertActive(false);
      setPaused(false);
    } catch (e) {
      console.error("Error stopping camera:", e);
    }
  }, []);

  // Auto-start when enabled changes
  useEffect(() => {
    if (enabled) {
      startCamera();
    } else {
      stopCamera();
    }

    return () => {
      stopCamera();
    };
    // startCamera and stopCamera are stable due to empty deps (TECH-6)
  }, [enabled, startCamera, stopCamera]);

  // Find primary camera
  const primaryCamera =
    Array.from(cameras.values()).find((c) => c.role === "primary") || null;

  return {
    cameras,
    primaryCamera,
    alertActive,
    paused,
    error,
    isStreaming,
    startCamera,
    stopCamera,
    getImageRef,
  };
}
