import { useState, useRef, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useCamera, type CameraState } from "@/hooks/useCamera";
import { LandmarkCanvas } from "@/components/LandmarkCanvas";
import { SignalsPanel } from "@/components/SignalsPanel";
import { StatusIndicator, type Status } from "@/components/StatusIndicator";
import { cn } from "@/lib/utils";
import {
  Camera,
  CameraOff,
  Eye,
  EyeOff,
  XCircle,
  AlertTriangle,
  Pause,
  Grid2x2,
  Maximize2,
} from "lucide-react";

export default function PreviewPage() {
  const [showLandmarks, setShowLandmarks] = useState(true);
  const [cameraEnabled, setCameraEnabled] = useState(true);
  const [viewMode, setViewMode] = useState<"primary" | "grid">("primary");

  const { cameras, primaryCamera, alertActive, paused, error, isStreaming, getImageRef } =
    useCamera({
      enabled: cameraEnabled,
    });

  const handleDismiss = async () => {
    try {
      await invoke("dismiss_alert");
    } catch (e) {
      console.error("Failed to dismiss:", e);
    }
  };

  const handleMarkMissed = async () => {
    try {
      await invoke("mark_missed_event");
    } catch (e) {
      console.error("Failed to mark missed:", e);
    }
  };

  const getStatus = (): Status => {
    if (!cameraEnabled || !isStreaming) return "offline";
    if (paused) return "offline";
    if (alertActive) return "alert";
    return "normal";
  };

  const cameraList = Array.from(cameras.values());
  const hasMultipleCameras = cameraList.length > 1;

  return (
    <div className="flex h-full flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h1 className="text-2xl font-bold">Camera Preview</h1>
          <StatusIndicator status={getStatus()} />
          {paused && (
            <span className="flex items-center gap-1 text-sm text-muted-foreground">
              <Pause className="h-4 w-4" />
              Paused
            </span>
          )}
          {hasMultipleCameras && (
            <span className="text-sm text-muted-foreground">
              {cameraList.length} cameras
            </span>
          )}
        </div>

        <div className="flex gap-2">
          {hasMultipleCameras && (
            <button
              onClick={() =>
                setViewMode(viewMode === "primary" ? "grid" : "primary")
              }
              className={cn(
                "flex items-center gap-2 rounded-lg px-3 py-2 text-sm transition-colors",
                "bg-secondary text-secondary-foreground hover:bg-secondary/80"
              )}
              title={
                viewMode === "primary"
                  ? "Show grid view"
                  : "Show primary camera only"
              }
            >
              {viewMode === "primary" ? (
                <Grid2x2 className="h-4 w-4" />
              ) : (
                <Maximize2 className="h-4 w-4" />
              )}
              {viewMode === "primary" ? "Grid" : "Primary"}
            </button>
          )}

          <button
            onClick={() => setShowLandmarks(!showLandmarks)}
            className={cn(
              "flex items-center gap-2 rounded-lg px-3 py-2 text-sm transition-colors",
              showLandmarks
                ? "bg-primary text-primary-foreground"
                : "bg-secondary text-secondary-foreground hover:bg-secondary/80"
            )}
            title={showLandmarks ? "Hide landmarks" : "Show landmarks"}
          >
            {showLandmarks ? (
              <Eye className="h-4 w-4" />
            ) : (
              <EyeOff className="h-4 w-4" />
            )}
            Landmarks
          </button>

          <button
            onClick={() => setCameraEnabled(!cameraEnabled)}
            className={cn(
              "flex items-center gap-2 rounded-lg px-3 py-2 text-sm transition-colors",
              cameraEnabled
                ? "bg-primary text-primary-foreground"
                : "bg-secondary text-secondary-foreground hover:bg-secondary/80"
            )}
          >
            {cameraEnabled ? (
              <Camera className="h-4 w-4" />
            ) : (
              <CameraOff className="h-4 w-4" />
            )}
            {cameraEnabled ? "Stop" : "Start"}
          </button>
        </div>
      </div>

      {/* Error display */}
      {error && (
        <div className="flex items-center gap-2 rounded-lg bg-destructive/10 p-4 text-destructive">
          <AlertTriangle className="h-5 w-5" />
          <span>{error}</span>
        </div>
      )}

      {/* Main content */}
      <div className="flex flex-1 gap-4">
        {/* Video feed */}
        <div className="relative flex-1">
          {isStreaming ? (
            viewMode === "primary" || !hasMultipleCameras ? (
              <CameraFeed
                camera={primaryCamera}
                getImageRef={getImageRef}
                showLandmarks={showLandmarks}
              />
            ) : (
              <div className="grid gap-2 grid-cols-2">
                {cameraList.map((camera) => (
                  <CameraFeed
                    key={camera.cameraId}
                    camera={camera}
                    getImageRef={getImageRef}
                    showLandmarks={showLandmarks}
                    compact
                  />
                ))}
              </div>
            )
          ) : (
            <div className="flex h-[480px] w-[640px] items-center justify-center rounded-lg bg-muted">
              <div className="flex flex-col items-center gap-2 text-muted-foreground">
                <CameraOff className="h-12 w-12" />
                <span>Camera off</span>
              </div>
            </div>
          )}

          {/* Alert overlay */}
          {alertActive && (
            <div className="absolute inset-0 flex items-center justify-center rounded-lg bg-red-500/20">
              <div className="rounded-lg bg-background/90 p-4 text-center">
                <p className="text-lg font-bold text-destructive">
                  BFRB Detected
                </p>
                <p className="text-sm text-muted-foreground">
                  Complete the exercise to dismiss
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Side panel */}
        <div className="flex w-64 flex-col gap-4">
          {/* Detection info */}
          <div className="rounded-lg border bg-card p-4">
            <h3 className="mb-2 font-semibold">Detection</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Hands:</span>
                <span>
                  {primaryCamera?.hands.length ?? 0}
                  {(primaryCamera?.hands.length ?? 0) > 0 && (
                    <span className="ml-1 text-xs text-muted-foreground">
                      (L:{primaryCamera?.hands.filter(h => h.handedness === "left").length ?? 0}
                      {" "}R:{primaryCamera?.hands.filter(h => h.handedness === "right").length ?? 0})
                    </span>
                  )}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Face:</span>
                <span>{primaryCamera?.face ? "Yes" : "No"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Pose:</span>
                <span>{primaryCamera?.pose ? "Yes" : "No"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Alert:</span>
                <span className={alertActive ? "text-destructive" : ""}>
                  {alertActive ? "Active" : "None"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Cameras:</span>
                <span>{cameraList.length}</span>
              </div>
            </div>
          </div>

          {/* Live signals — shows each detector's current "why" */}
          <SignalsPanel
            title="Signals"
            explanations={primaryCamera?.currentSignals ?? []}
          />

          {/* Camera list */}
          {hasMultipleCameras && (
            <div className="rounded-lg border bg-card p-4">
              <h3 className="mb-2 font-semibold">Cameras</h3>
              <div className="space-y-2 text-sm">
                {cameraList.map((camera) => (
                  <div
                    key={camera.cameraId}
                    className="flex items-center justify-between"
                  >
                    <span className="text-muted-foreground">
                      {camera.cameraId}
                    </span>
                    <span
                      className={cn(
                        "rounded px-1.5 py-0.5 text-xs",
                        camera.role === "primary"
                          ? "bg-primary/20 text-primary"
                          : "bg-muted text-muted-foreground"
                      )}
                    >
                      {camera.role}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Quick actions */}
          <div className="rounded-lg border bg-card p-4">
            <h3 className="mb-2 font-semibold">Quick Actions</h3>
            <div className="flex flex-col gap-2">
              <button
                onClick={handleDismiss}
                className="flex items-center justify-center gap-2 rounded-lg bg-secondary px-3 py-2 text-sm transition-colors hover:bg-secondary/80"
                disabled={!alertActive}
              >
                <XCircle className="h-4 w-4" />
                Dismiss Alert (F9)
              </button>
              <button
                onClick={handleMarkMissed}
                className="flex items-center justify-center gap-2 rounded-lg bg-secondary px-3 py-2 text-sm transition-colors hover:bg-secondary/80"
              >
                <AlertTriangle className="h-4 w-4" />
                Mark Missed (F10)
              </button>
            </div>
          </div>

          {/* Hotkey reference */}
          <div className="rounded-lg border bg-card p-4">
            <h3 className="mb-2 font-semibold">Hotkeys</h3>
            <div className="space-y-1 text-sm text-muted-foreground">
              <div>
                <kbd className="rounded bg-muted px-1">F9</kbd> Dismiss
              </div>
              <div>
                <kbd className="rounded bg-muted px-1">F10</kbd> Mark missed
              </div>
              <div>
                <kbd className="rounded bg-muted px-1">F11</kbd> Pause/Resume
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

interface CameraFeedProps {
  camera: CameraState | null;
  getImageRef: (cameraId: string) => React.RefObject<HTMLImageElement | null>;
  showLandmarks: boolean;
  compact?: boolean;
}

function CameraFeed({
  camera,
  getImageRef,
  showLandmarks,
  compact = false,
}: CameraFeedProps) {
  const imageRef = useRef<HTMLImageElement | null>(null);
  const cameraId = camera?.cameraId ?? "unknown";

  // Sync with the hook's image ref
  useEffect(() => {
    const hookRef = getImageRef(cameraId);
    hookRef.current = imageRef.current;
  }, [cameraId, getImageRef]);

  if (!camera) {
    return (
      <div
        className={cn(
          "flex items-center justify-center rounded-lg bg-muted",
          compact ? "h-[240px] w-[320px]" : "h-[480px] w-[640px]"
        )}
      >
        <div className="flex flex-col items-center gap-2 text-muted-foreground">
          <CameraOff className="h-8 w-8" />
          <span>No camera</span>
        </div>
      </div>
    );
  }

  // Use actual camera dimensions, scaled down if compact mode
  const scale = compact ? 0.5 : 1;
  const width = Math.round((camera.width || 640) * scale);
  const height = Math.round((camera.height || 480) * scale);

  return (
    <div className="relative">
      {/* Hidden image element that receives frame data */}
      <img
        ref={imageRef}
        alt={`Camera ${camera.cameraId}`}
        className="hidden"
        width={width}
        height={height}
      />

      <LandmarkCanvas
        width={width}
        height={height}
        hands={camera.hands}
        face={camera.face}
        pose={camera.pose}
        imageRef={imageRef}
        showLandmarks={showLandmarks}
        frameBase64={camera.frameBase64}
        signals={camera.currentSignals}
      />

      {/* Camera label for grid view */}
      {compact && (
        <div className="absolute left-2 top-2 rounded bg-background/80 px-2 py-1 text-xs font-medium">
          {camera.cameraId}
          <span
            className={cn(
              "ml-2 rounded px-1",
              camera.role === "primary"
                ? "bg-primary/20 text-primary"
                : "bg-muted text-muted-foreground"
            )}
          >
            {camera.role}
          </span>
        </div>
      )}
    </div>
  );
}
