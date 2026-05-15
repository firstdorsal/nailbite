import { useState, useRef, useEffect } from "react";
import { useCamera, type CameraState } from "@/hooks/useCamera";
import { LandmarkCanvas } from "@/components/LandmarkCanvas";
import { cn } from "@/lib/utils";
import {
  CameraOff,
  Eye,
  EyeOff,
  AlertTriangle,
  Grid2x2,
  Maximize2,
} from "lucide-react";

export default function PreviewPage() {
  const [showLandmarks, setShowLandmarks] = useState(true);
  const [viewMode, setViewMode] = useState<"primary" | "grid">("primary");

  const { cameras, primaryCamera, error, isStreaming, getImageRef } =
    useCamera({ enabled: true });

  const cameraList = Array.from(cameras.values());
  const hasMultipleCameras = cameraList.length > 1;

  return (
    <div className="flex h-full flex-col gap-4">
      {/* Error display */}
      {error && (
        <div className="flex items-center gap-2 rounded-lg bg-destructive/10 p-4 text-destructive">
          <AlertTriangle className="h-5 w-5" />
          <span>{error}</span>
        </div>
      )}

      {/* Camera + signals row. Both columns share a fixed height
          (PREVIEW_ROW_HEIGHT) so the camera is visually the same
          size as the panel beside it. We deliberately do NOT use
          `flex-1 items-stretch` here — the row would inherit the
          whole page height, the canvas's aspect-ratio CSS would
          then compute a very wide width, and the camera column
          would overflow and slide under the (transparent) signals
          column. A bounded height keeps both bounded. */}
      <div className="flex h-[540px] items-start gap-4">
        <div className="flex h-full items-start gap-2">
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
            <div className="flex h-full aspect-[4/3] items-center justify-center rounded-lg bg-muted">
              <div className="flex flex-col items-center gap-2 text-muted-foreground">
                <CameraOff className="h-12 w-12" />
                <span>Camera off</span>
              </div>
            </div>
          )}

          {/* Compact icon-button column directly attached to the camera. */}
          <div className="flex flex-col gap-2">
            <CameraIconButton
              onClick={() => setShowLandmarks(!showLandmarks)}
              active={showLandmarks}
              title={showLandmarks ? "Hide landmarks" : "Show landmarks"}
            >
              {showLandmarks ? (
                <Eye className="h-4 w-4" />
              ) : (
                <EyeOff className="h-4 w-4" />
              )}
            </CameraIconButton>
            {hasMultipleCameras && (
              <CameraIconButton
                onClick={() =>
                  setViewMode(viewMode === "primary" ? "grid" : "primary")
                }
                active={viewMode === "grid"}
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
              </CameraIconButton>
            )}
          </div>
        </div>

      </div>
    </div>
  );
}

interface CameraIconButtonProps {
  onClick: () => void;
  active: boolean;
  title: string;
  children: React.ReactNode;
}

function CameraIconButton({
  onClick,
  active,
  title,
  children,
}: CameraIconButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      aria-label={title}
      className={cn(
        "flex h-8 w-8 items-center justify-center rounded-lg transition-colors",
        active
          ? "bg-primary text-primary-foreground"
          : "bg-secondary text-secondary-foreground hover:bg-secondary/80",
      )}
    >
      {children}
    </button>
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
          compact ? "h-[240px] aspect-[4/3]" : "h-full aspect-[4/3]",
        )}
      >
        <div className="flex flex-col items-center gap-2 text-muted-foreground">
          <CameraOff className="h-8 w-8" />
          <span>No camera</span>
        </div>
      </div>
    );
  }

  // Canvas keeps its native capture resolution so landmark drawing
  // stays sharp; CSS scales it to fit the available container height
  // (preserving the 4:3 aspect ratio) so the preview is sized to
  // match the side panel rather than being locked to a fixed 480 px.
  const width = camera.width || 640;
  const height = camera.height || 480;

  return (
    <div
      className={cn(
        "relative",
        compact ? "h-[240px]" : "h-full aspect-[4/3]",
      )}
    >
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
        className="h-full w-full"
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
