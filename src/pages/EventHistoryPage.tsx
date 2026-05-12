import { useState, useEffect, useCallback, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  RefreshCw,
  AlertTriangle,
  Trash2,
  ChevronLeft,
  ChevronRight,
  Check,
  Eye,
  Hand,
  HelpCircle,
  ScanFace,
  PersonStanding,
  Clock,
  ImageIcon,
  ThumbsDown,
  ThumbsUp,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Slider } from "@/components/ui/slider";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DailyDetectionBars,
  dayKeyOf,
} from "@/components/DailyDetectionBars";
import { SignalsPanel } from "@/components/SignalsPanel";
import { SignalTimeline } from "@/components/SignalTimeline";
import { Input } from "@/components/ui/input";
import type {
  EventHistorySummary,
  EventHistoryDetail,
  FrameInfo,
  FrameOverlay,
  Verdict,
} from "@/types";
import { cn } from "@/lib/utils";

// MediaPipe hand-landmark skeleton edges (21 landmarks).
const HAND_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];
const OUTER_LIP_INDICES = [
  61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84,
  181, 91, 146,
];
const POSE_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 9], [0, 10], [9, 10],
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
];
const POSE_VISIBILITY_THRESHOLD = 0.5;

/** Draw the overlay onto a canvas sized to match the image's natural
 *  resolution. Coordinates are normalized [0,1] → multiplied by w/h. */
function drawOverlay(
  ctx: CanvasRenderingContext2D,
  overlay: FrameOverlay,
  w: number,
  h: number,
) {
  ctx.clearRect(0, 0, w, h);

  for (const hand of overlay.hands) {
    const color =
      hand.handedness === "left"
        ? "#00ff00"
        : hand.handedness === "right"
          ? "#ff6600"
          : "#888";
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    for (const [a, b] of HAND_CONNECTIONS) {
      const la = hand.landmarks[a];
      const lb = hand.landmarks[b];
      if (!la || !lb) continue;
      ctx.beginPath();
      ctx.moveTo(la.x * w, la.y * h);
      ctx.lineTo(lb.x * w, lb.y * h);
      ctx.stroke();
    }
    ctx.fillStyle = color;
    for (const lm of hand.landmarks) {
      ctx.beginPath();
      ctx.arc(lm.x * w, lm.y * h, 3, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  if (overlay.face) {
    ctx.strokeStyle = "#ff00ff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    OUTER_LIP_INDICES.forEach((idx, i) => {
      const lm = overlay.face!.landmarks[idx];
      if (!lm) return;
      if (i === 0) ctx.moveTo(lm.x * w, lm.y * h);
      else ctx.lineTo(lm.x * w, lm.y * h);
    });
    ctx.closePath();
    ctx.stroke();
  }

  if (overlay.pose) {
    const pose = overlay.pose;
    ctx.strokeStyle = "#00ffff";
    ctx.lineWidth = 2;
    for (const [a, b] of POSE_CONNECTIONS) {
      const la = pose.landmarks[a];
      const lb = pose.landmarks[b];
      if (!la || !lb) continue;
      if (
        la.visibility < POSE_VISIBILITY_THRESHOLD ||
        lb.visibility < POSE_VISIBILITY_THRESHOLD
      ) {
        continue;
      }
      ctx.beginPath();
      ctx.moveTo(la.x * w, la.y * h);
      ctx.lineTo(lb.x * w, lb.y * h);
      ctx.stroke();
    }
    ctx.fillStyle = "#00ffff";
    for (let i = 0; i < Math.min(pose.landmarks.length, 17); i += 1) {
      const lm = pose.landmarks[i];
      if (!lm || lm.visibility < POSE_VISIBILITY_THRESHOLD) continue;
      ctx.beginPath();
      ctx.arc(lm.x * w, lm.y * h, 3, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}

function formatTrigger(trigger: string): string {
  switch (trigger) {
    case "detection":
      return "Detection";
    case "missed_event":
      return "Missed Event";
    case "false_positive":
      return "False Positive";
    default:
      return trigger;
  }
}

function triggerVariant(
  trigger: string
): "default" | "secondary" | "destructive" | "outline" {
  switch (trigger) {
    case "detection":
      return "destructive";
    case "missed_event":
      return "default";
    case "false_positive":
      return "secondary";
    default:
      return "outline";
  }
}

function formatBfrbType(bfrbType: string | null): string {
  if (!bfrbType) return "";
  return bfrbType
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function formatTimestamp(timestamp: string): string {
  try {
    const date = new Date(timestamp);
    return date.toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return timestamp;
  }
}

function formatFullTimestamp(timestamp: string): string {
  try {
    const date = new Date(timestamp);
    return date.toLocaleString();
  } catch {
    return timestamp;
  }
}

function ratingColor(rating: number | null): string {
  if (rating == null) return "text-muted-foreground";
  if (rating <= 2) return "text-destructive";
  if (rating <= 3) return "text-yellow-500";
  return "text-green-500";
}

/** Compact rating display for cards */
function RatingDisplay({ rating }: { rating: number | null }) {
  if (rating == null) return null;
  return (
    <Badge variant="outline" className={ratingColor(rating)}>
      {rating}/5
    </Badge>
  );
}

const VERDICT_LABEL: Record<Verdict, string> = {
  true_positive: "Correct",
  false_positive: "False positive",
  unsure: "Unsure",
};

const VERDICT_BADGE_CLASS: Record<Verdict, string> = {
  true_positive: "border-green-500/40 bg-green-500/10 text-green-600",
  false_positive: "border-red-500/40 bg-red-500/10 text-red-600",
  unsure: "border-muted-foreground/40 bg-muted text-muted-foreground",
};

const UNLABELED_BADGE_CLASS =
  "border-muted-foreground/30 bg-transparent text-muted-foreground";

function VerdictBadge({ verdict }: { verdict: Verdict | null }) {
  if (!verdict) {
    return (
      <Badge variant="outline" className={UNLABELED_BADGE_CLASS}>
        Unlabeled
      </Badge>
    );
  }
  return (
    <Badge variant="outline" className={VERDICT_BADGE_CLASS[verdict]}>
      {VERDICT_LABEL[verdict]}
    </Badge>
  );
}

interface VerdictPillsProps {
  value: Verdict | null;
  reason: string;
  busy: boolean;
  onChangeVerdict: (verdict: Verdict | null) => void;
  onChangeReason: (reason: string) => void;
  onCommitReason: () => void;
}

function VerdictPills({
  value,
  reason,
  busy,
  onChangeVerdict,
  onChangeReason,
  onCommitReason,
}: VerdictPillsProps) {
  const buttons: Array<{
    verdict: Verdict;
    label: string;
    icon: React.ReactNode;
  }> = [
    {
      verdict: "true_positive",
      label: "True positive",
      icon: <Check className="h-3.5 w-3.5" />,
    },
    {
      verdict: "false_positive",
      label: "False positive",
      icon: <X className="h-3.5 w-3.5" />,
    },
    {
      verdict: "unsure",
      label: "Unsure",
      icon: <HelpCircle className="h-3.5 w-3.5" />,
    },
  ];

  return (
    <div className="space-y-2 rounded-md border bg-muted/40 p-3">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium">Verdict</span>
        {value && (
          <Button
            variant="ghost"
            size="sm"
            className="h-6 px-2 text-xs text-muted-foreground"
            onClick={() => onChangeVerdict(null)}
            disabled={busy}
          >
            Clear
          </Button>
        )}
      </div>
      <div className="grid grid-cols-3 gap-2">
        {buttons.map((b) => {
          const active = value === b.verdict;
          return (
            <button
              key={b.verdict}
              type="button"
              onClick={() => onChangeVerdict(b.verdict)}
              disabled={busy}
              className={cn(
                "flex items-center justify-center gap-1.5 rounded border px-2 py-1.5 text-xs transition-colors",
                active
                  ? VERDICT_BADGE_CLASS[b.verdict]
                  : "border-border bg-background hover:bg-accent",
                busy && "opacity-60",
              )}
            >
              {b.icon}
              {b.label}
            </button>
          );
        })}
      </div>
      <Input
        value={reason}
        onChange={(e) => onChangeReason(e.target.value)}
        onBlur={onCommitReason}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.currentTarget.blur();
          }
        }}
        placeholder="Optional reason (e.g. 'phone in hand', 'real bite')"
        className="h-8 text-xs"
        disabled={busy}
      />
    </div>
  );
}

/** Rating slider control */
function RatingSlider({
  value,
  onChange,
}: {
  value: number | null;
  onChange: (rating: number | null) => void;
}) {
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-muted-foreground whitespace-nowrap">
        Rating:
      </span>
      <input
        type="range"
        min={1}
        max={5}
        step={1}
        value={value ?? 3}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-2 w-32 cursor-pointer appearance-none rounded-full bg-secondary accent-primary"
      />
      <span className={`min-w-[2rem] text-sm font-medium ${ratingColor(value)}`}>
        {value != null ? `${value}/5` : "---"}
      </span>
      {value != null && (
        <Button
          variant="ghost"
          size="sm"
          className="h-6 px-2 text-xs text-muted-foreground"
          onClick={(e) => {
            e.stopPropagation();
            onChange(null);
          }}
        >
          Clear
        </Button>
      )}
    </div>
  );
}

/** Thumbnail image that loads lazily */
function ThumbnailImage({
  eventId,
  filename,
}: {
  eventId: string;
  filename: string;
}) {
  const [imageData, setImageData] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);

    invoke<string>("get_event_frame", { eventId, filename })
      .then((base64) => {
        if (!cancelled) {
          setImageData(base64);
          setLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [eventId, filename]);

  if (loading) return <Skeleton className="h-full w-full" />;
  if (!imageData) {
    return (
      <div className="flex h-full items-center justify-center text-muted-foreground">
        <ImageIcon className="h-4 w-4" />
      </div>
    );
  }

  return (
    <img
      src={`data:image/jpeg;base64,${imageData}`}
      alt="Trigger frame"
      className="h-full w-full object-cover"
    />
  );
}

/** Frame viewer with base64 image loading. Two paths:
 *  - If a vector `overlay` is present, always fetch the RAW jpeg and draw
 *    the overlay on a canvas above it (toggled by `showAnnotated`).
 *  - Otherwise fall back to the legacy baked `annotated_filename` jpeg
 *    so historic events still render their annotations. */
function FrameViewer({
  eventId,
  frame,
  showAnnotated,
}: {
  eventId: string;
  frame: FrameInfo;
  showAnnotated: boolean;
}) {
  const [imageData, setImageData] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const hasVectorOverlay = frame.overlay != null;
  // When we have vector overlay data, always fetch the raw jpeg; the canvas
  // does the rendering. Otherwise fall back to whatever the toggle picks.
  const filename = hasVectorOverlay
    ? frame.raw_filename
    : showAnnotated && frame.annotated_filename
      ? frame.annotated_filename
      : frame.raw_filename;

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setImageData(null);

    invoke<string>("get_event_frame", { eventId, filename })
      .then((base64) => {
        if (!cancelled) {
          setImageData(base64);
          setLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [eventId, filename]);

  // Draw the vector overlay onto the canvas once the image has loaded.
  // Canvas's intrinsic resolution = image's natural resolution and both
  // share `object-contain` CSS, so they letterbox identically inside the
  // 4:3 container regardless of the source camera aspect ratio.
  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    const overlay = frame.overlay;
    if (!canvas || !img || !showAnnotated || !overlay) {
      if (canvas) {
        const ctx = canvas.getContext("2d");
        ctx?.clearRect(0, 0, canvas.width, canvas.height);
      }
      return;
    }
    const draw = () => {
      const w = img.naturalWidth;
      const h = img.naturalHeight;
      if (!w || !h) return;
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      drawOverlay(ctx, overlay, w, h);
    };
    if (img.complete && img.naturalWidth > 0) {
      draw();
    } else {
      img.addEventListener("load", draw, { once: true });
      return () => img.removeEventListener("load", draw);
    }
  }, [imageData, frame.overlay, showAnnotated]);

  return (
    <div className="space-y-2">
      <div className="relative aspect-[4/3] overflow-hidden rounded-md border bg-muted">
        {loading ? (
          <Skeleton className="h-full w-full" />
        ) : imageData ? (
          <>
            <img
              ref={imgRef}
              src={`data:image/jpeg;base64,${imageData}`}
              alt={`Frame ${frame.offset >= 0 ? "+" : ""}${frame.offset}`}
              className="absolute inset-0 h-full w-full object-contain"
            />
            {hasVectorOverlay && (
              <canvas
                ref={canvasRef}
                className={cn(
                  "pointer-events-none absolute inset-0 h-full w-full object-contain",
                  !showAnnotated && "hidden",
                )}
              />
            )}
          </>
        ) : (
          <div className="flex h-full items-center justify-center text-muted-foreground">
            <ImageIcon className="mr-2 h-5 w-5" />
            Failed to load
          </div>
        )}
      </div>

      <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
        <span className="font-mono">
          {frame.offset >= 0 ? "+" : ""}
          {frame.offset}
        </span>
        {frame.hand_count > 0 && (
          <span className="flex items-center gap-1">
            <Hand className="h-3 w-3" />
            {frame.hand_sides.join(", ")}
          </span>
        )}
        {frame.face_detected && (
          <span className="flex items-center gap-1">
            <ScanFace className="h-3 w-3" />
            Face
          </span>
        )}
        {frame.pose_detected && (
          <span className="flex items-center gap-1">
            <PersonStanding className="h-3 w-3" />
            Pose
          </span>
        )}
      </div>
    </div>
  );
}

/** Event detail dialog */
function EventDetailView({
  eventId,
  open,
  onClose,
  onRate,
  onVerdict,
}: {
  eventId: string | null;
  open: boolean;
  onClose: () => void;
  onRate: (eventId: string, rating: number | null) => void;
  onVerdict: (
    eventId: string,
    verdict: Verdict | null,
    reason: string | null,
  ) => Promise<void>;
}) {
  const [detail, setDetail] = useState<EventHistoryDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [showAnnotated, setShowAnnotated] = useState(true);
  const [reasonDraft, setReasonDraft] = useState("");
  const [verdictBusy, setVerdictBusy] = useState(false);

  useEffect(() => {
    if (!eventId || !open) return;
    let cancelled = false;
    setLoading(true);
    setCurrentFrame(0);

    invoke<EventHistoryDetail>("get_event_details", { eventId })
      .then((result) => {
        if (!cancelled) {
          setDetail(result);
          setReasonDraft(result.verdict_reason ?? "");
          setLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [eventId, open]);

  const frame = detail?.frames[currentFrame];
  const hasAnnotated = frame?.annotated_filename != null;
  const totalFrames = detail?.frames.length ?? 0;

  const handleRate = (rating: number | null) => {
    if (!detail) return;
    setDetail({ ...detail, user_rating: rating });
    onRate(detail.id, rating);
  };

  const persistVerdict = async (
    nextVerdict: Verdict | null,
    nextReason: string | null,
  ) => {
    if (!detail) return;
    setVerdictBusy(true);
    try {
      await onVerdict(detail.id, nextVerdict, nextReason);
      setDetail({
        ...detail,
        verdict: nextVerdict,
        verdict_reason: nextReason ?? null,
      });
    } finally {
      setVerdictBusy(false);
    }
  };

  const handleVerdictChange = (next: Verdict | null) => {
    void persistVerdict(next, reasonDraft.trim() || null);
  };

  const handleReasonBlur = () => {
    if (!detail) return;
    const trimmed = reasonDraft.trim();
    const current = detail.verdict_reason ?? "";
    if (trimmed === current) return;
    void persistVerdict(detail.verdict, trimmed || null);
  };

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="flex max-h-[90vh] max-w-3xl flex-col overflow-hidden lg:max-w-5xl xl:max-w-6xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            {detail && (
              <>
                <Badge variant={triggerVariant(detail.trigger)}>
                  {formatTrigger(detail.trigger)}
                </Badge>
                {detail.bfrb_type && (
                  <span>{formatBfrbType(detail.bfrb_type)}</span>
                )}
                <VerdictBadge verdict={detail.verdict} />
                <RatingDisplay rating={detail.user_rating} />
              </>
            )}
          </DialogTitle>
          <DialogDescription>
            {detail && (
              <span className="flex items-center gap-2">
                <Clock className="h-3 w-3" />
                {formatFullTimestamp(detail.timestamp)}
                {detail.confidence != null && (
                  <>
                    <Separator orientation="vertical" className="h-3" />
                    Confidence: {Math.round(detail.confidence * 100)}%
                  </>
                )}
              </span>
            )}
          </DialogDescription>
        </DialogHeader>

        {loading ? (
          <div className="space-y-4 p-4">
            <Skeleton className="h-64 w-full" />
            <div className="flex gap-2">
              <Skeleton className="h-8 w-20" />
              <Skeleton className="h-8 w-20" />
            </div>
          </div>
        ) : detail && frame ? (
          <div className="-mr-2 min-h-0 flex-1 space-y-4 overflow-y-auto pr-2">
            <FrameViewer
              eventId={detail.id}
              frame={frame}
              showAnnotated={showAnnotated}
            />

            {/* Controls */}
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    setCurrentFrame((prev) => Math.max(0, prev - 1))
                  }
                  disabled={currentFrame === 0}
                >
                  <ChevronLeft className="h-4 w-4" />
                </Button>
                <span className="min-w-[5rem] text-center text-sm text-muted-foreground">
                  {currentFrame + 1} / {totalFrames}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    setCurrentFrame((prev) =>
                      Math.min(totalFrames - 1, prev + 1)
                    )
                  }
                  disabled={currentFrame >= totalFrames - 1}
                >
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>

              {hasAnnotated && (
                <Button
                  variant={showAnnotated ? "default" : "outline"}
                  size="sm"
                  onClick={() => setShowAnnotated((prev) => !prev)}
                >
                  <Eye className="mr-1 h-4 w-4" />
                  {showAnnotated ? "Annotated" : "Raw"}
                </Button>
              )}
            </div>

            {/* Verdict + reason */}
            <VerdictPills
              value={detail.verdict}
              reason={reasonDraft}
              busy={verdictBusy}
              onChangeVerdict={handleVerdictChange}
              onChangeReason={setReasonDraft}
              onCommitReason={handleReasonBlur}
            />

            {/* Rating slider */}
            <div className="flex items-center justify-between rounded-md border bg-muted/50 px-3 py-2">
              <RatingSlider
                value={detail.user_rating}
                onChange={handleRate}
              />
            </div>

            {/* Signal timeline (proximity per detector across event) */}
            <SignalTimeline
              frames={detail.frames}
              activeIndex={currentFrame}
              onSelect={setCurrentFrame}
            />

            {/* Per-frame signal breakdown */}
            {frame.explanations.length > 0 && (
              <div className="rounded-md border bg-muted/30 p-3">
                <SignalsPanel
                  inline
                  explanations={frame.explanations}
                />
              </div>
            )}

            {/* Frame timeline.
                - Narrow (<lg): compact ±N offset buttons.
                - Wide (≥lg): actual image thumbnails of every frame side
                  by side, so the user can see the whole gesture at once. */}
            <ScrollArea className="w-full">
              <div className="flex gap-1 pb-2 lg:hidden">
                {detail.frames.map((f, i) => (
                  <button
                    key={f.offset}
                    onClick={() => setCurrentFrame(i)}
                    className={`flex h-8 min-w-[2.5rem] items-center justify-center rounded border text-xs font-mono transition-colors ${
                      i === currentFrame
                        ? "border-primary bg-primary text-primary-foreground"
                        : f.offset === 0
                          ? "border-destructive/50 bg-destructive/10 text-destructive hover:bg-destructive/20"
                          : "border-border bg-card text-muted-foreground hover:bg-accent"
                    }`}
                  >
                    {f.offset >= 0 ? "+" : ""}
                    {f.offset}
                  </button>
                ))}
              </div>
              <div className="hidden gap-2 pb-2 lg:flex">
                {detail.frames.map((f, i) => (
                  <button
                    key={f.offset}
                    type="button"
                    onClick={() => setCurrentFrame(i)}
                    title={`Frame ${f.offset >= 0 ? "+" : ""}${f.offset}`}
                    className={cn(
                      "relative h-24 w-32 shrink-0 overflow-hidden rounded border transition-all",
                      i === currentFrame
                        ? "border-primary ring-2 ring-primary"
                        : f.offset === 0
                          ? "border-destructive/60 hover:border-destructive"
                          : "border-border hover:border-foreground/30",
                    )}
                  >
                    <ThumbnailImage
                      eventId={detail.id}
                      filename={f.raw_filename}
                    />
                    <span
                      className={cn(
                        "absolute bottom-0 right-0 rounded-tl px-1 text-[10px] font-mono leading-tight text-white",
                        f.offset === 0 ? "bg-destructive/80" : "bg-black/70",
                      )}
                    >
                      {f.offset >= 0 ? "+" : ""}
                      {f.offset}
                    </span>
                  </button>
                ))}
              </div>
            </ScrollArea>
          </div>
        ) : (
          <div className="flex items-center justify-center p-8 text-muted-foreground">
            No event data available
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

/** Width range for the row thumbnail (px). Height stays 4:3. */
const THUMB_MIN = 96;
const THUMB_MAX = 384;
const THUMB_STORAGE_KEY = "nailbite.event-history.thumb-width";

function todayKey(): string {
  return dayKeyOf(new Date());
}

export default function EventHistoryPage() {
  const [events, setEvents] = useState<EventHistorySummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedEvent, setSelectedEvent] = useState<string | null>(null);
  const [detailOpen, setDetailOpen] = useState(false);
  // Page opens filtered to today by default. Click a bar to switch days,
  // click "All" to clear the filter.
  const [selectedDate, setSelectedDate] = useState<string | null>(todayKey);
  // Thumbnail size, persisted to localStorage so the user's preference
  // survives reloads. Starts at 192 (2× the original h-24 w-32).
  const [thumbWidth, setThumbWidth] = useState<number>(() => {
    if (typeof window === "undefined") return 192;
    const stored = window.localStorage.getItem(THUMB_STORAGE_KEY);
    const parsed = stored ? Number(stored) : NaN;
    if (Number.isFinite(parsed) && parsed >= THUMB_MIN && parsed <= THUMB_MAX) {
      return parsed;
    }
    return 192;
  });

  useEffect(() => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem(THUMB_STORAGE_KEY, String(thumbWidth));
    }
  }, [thumbWidth]);

  const loadEvents = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const result =
        await invoke<EventHistorySummary[]>("list_event_history");
      setEvents(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadEvents();
  }, [loadEvents]);

  const handleViewEvent = (eventId: string) => {
    setSelectedEvent(eventId);
    setDetailOpen(true);
  };

  const handleRateEvent = async (
    eventId: string,
    rating: number | null
  ) => {
    try {
      await invoke("rate_event", { eventId, rating });
      setEvents((prev) =>
        prev.map((ev) =>
          ev.id === eventId ? { ...ev, user_rating: rating } : ev
        )
      );
    } catch (err) {
      console.error("Failed to rate event:", err);
    }
  };

  const handleVerdictEvent = async (
    eventId: string,
    verdict: Verdict | null,
    reason: string | null,
  ) => {
    try {
      await invoke("set_event_verdict", { eventId, verdict, reason });
      setEvents((prev) =>
        prev.map((ev) =>
          ev.id === eventId ? { ...ev, verdict } : ev,
        ),
      );
    } catch (err) {
      console.error("Failed to set verdict:", err);
    }
  };

  const handleDeleteEvent = async (
    e: React.MouseEvent,
    eventId: string
  ) => {
    e.stopPropagation();
    try {
      await invoke("delete_event", { eventId });
      setEvents((prev) => prev.filter((ev) => ev.id !== eventId));
    } catch (err) {
      console.error("Failed to delete event:", err);
    }
  };

  if (loading) {
    return (
      <div className="flex h-full flex-col gap-4">
        <div className="flex items-center justify-between">
          <Skeleton className="h-8 w-48" />
          <Skeleton className="h-9 w-24" />
        </div>
        <div className="grid gap-3">
          {[1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-28 w-full" />
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-4">
        <AlertTriangle className="h-12 w-12 text-destructive" />
        <p className="text-destructive">{error}</p>
        <Button onClick={loadEvents} variant="outline">
          Retry
        </Button>
      </div>
    );
  }

  // Apply the day filter (if any) before rendering and counting.
  const filteredEvents = selectedDate
    ? events.filter((e) => {
        const d = new Date(e.timestamp);
        if (Number.isNaN(d.getTime())) return false;
        return dayKeyOf(d) === selectedDate;
      })
    : events;

  const counts = filteredEvents.reduce(
    (acc, e) => {
      if (e.verdict === "true_positive") acc.tp += 1;
      else if (e.verdict === "false_positive") acc.fp += 1;
      else if (e.verdict === "unsure") acc.unsure += 1;
      else acc.unlabeled += 1;
      return acc;
    },
    { tp: 0, fp: 0, unsure: 0, unlabeled: 0 },
  );

  return (
    <div className="flex h-full flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold">Event History</h1>
          <p className="text-sm text-muted-foreground">
            {selectedDate
              ? `${filteredEvents.length} on ${selectedDate}${
                  selectedDate === todayKey() ? " (today)" : ""
                } — ${events.length} total`
              : `${events.length} recorded event${events.length !== 1 ? "s" : ""}`}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <ImageIcon className="h-4 w-4 text-muted-foreground" />
            <Slider
              value={[thumbWidth]}
              min={THUMB_MIN}
              max={THUMB_MAX}
              step={16}
              onValueChange={([v]) => setThumbWidth(v ?? thumbWidth)}
              className="w-32"
              aria-label="Thumbnail size"
            />
          </div>
          <Button onClick={loadEvents} variant="outline" size="sm">
            <RefreshCw className="mr-2 h-4 w-4" />
            Refresh
          </Button>
        </div>
      </div>

      {events.length > 0 && (
        <DailyDetectionBars
          events={events}
          selectedDate={selectedDate}
          onSelectDate={setSelectedDate}
        />
      )}

      {events.length > 0 && (
        <div className="flex flex-wrap gap-2 text-xs">
          <Badge variant="outline" className={VERDICT_BADGE_CLASS.true_positive}>
            <Check className="mr-1 h-3 w-3" />
            {counts.tp} correct
          </Badge>
          <Badge variant="outline" className={VERDICT_BADGE_CLASS.false_positive}>
            <X className="mr-1 h-3 w-3" />
            {counts.fp} false positive
          </Badge>
          <Badge variant="outline" className={VERDICT_BADGE_CLASS.unsure}>
            <HelpCircle className="mr-1 h-3 w-3" />
            {counts.unsure} unsure
          </Badge>
          <Badge variant="outline" className="text-muted-foreground">
            {counts.unlabeled} unlabeled
          </Badge>
        </div>
      )}

      <Separator />

      {/* Event list */}
      {events.length === 0 ? (
        <div className="flex flex-1 flex-col items-center justify-center gap-3 text-muted-foreground">
          <ImageIcon className="h-12 w-12" />
          <p>No events recorded yet</p>
          <p className="text-xs">
            Events are captured automatically when behaviors are detected
          </p>
        </div>
      ) : filteredEvents.length === 0 ? (
        <div className="flex flex-1 flex-col items-center justify-center gap-3 text-muted-foreground">
          <ImageIcon className="h-12 w-12" />
          <p>No events on {selectedDate}</p>
          <Button variant="outline" size="sm" onClick={() => setSelectedDate(null)}>
            Show all days
          </Button>
        </div>
      ) : (
        <ScrollArea className="flex-1">
          <div className="grid gap-3 pr-4">
            {filteredEvents.map((event) => {
              const thumbFilename =
                event.trigger_frame_annotated ?? event.trigger_frame;
              // Pick up to 5 keyframes evenly spaced through the event, so
              // wide rows can show a filmstrip instead of a single thumb.
              // Always include the first, last, and trigger frames.
              const allFrames = event.frame_files;
              const keyframes: string[] = (() => {
                if (allFrames.length <= 5) return allFrames;
                const triggerIdx = Math.max(
                  0,
                  allFrames.findIndex((f) => f.includes("+00")),
                );
                const picks = new Set<number>([0, triggerIdx, allFrames.length - 1]);
                // Fill out to 5 with evenly-spaced indices.
                for (let i = 1; i < 4 && picks.size < 5; i += 1) {
                  picks.add(Math.round((i * (allFrames.length - 1)) / 4));
                }
                return [...picks]
                  .sort((a, b) => a - b)
                  .map((i) => allFrames[i])
                  .filter((f): f is string => f != null);
              })();
              const filmStripHeight = Math.round(thumbWidth * 0.75);

              return (
                <Card
                  key={event.id}
                  className="cursor-pointer transition-colors hover:bg-accent/50"
                  onClick={() => handleViewEvent(event.id)}
                >
                  <div className="flex">
                    {/* Content — description on the left, images on the right. */}
                    <div className="flex-1 min-w-0">
                      <CardHeader className="p-3 pb-1">
                        <div className="flex items-start justify-between gap-2">
                          <div className="flex flex-wrap items-center gap-1.5">
                            {event.bfrb_type && (
                              <CardTitle className="text-sm">
                                {formatBfrbType(event.bfrb_type)}
                              </CardTitle>
                            )}
                            <VerdictBadge verdict={event.verdict} />
                            <RatingDisplay rating={event.user_rating} />
                          </div>
                          <div className="flex shrink-0 items-center gap-1">
                            {/* Quick-verdict buttons — mark true/false
                                positive without opening the detail dialog.
                                Re-clicking the active verdict clears it. */}
                            <Button
                              variant="ghost"
                              size="icon"
                              className={cn(
                                "h-7 w-7 text-muted-foreground hover:bg-green-500/10 hover:text-green-600",
                                event.verdict === "true_positive" &&
                                  "bg-green-500/15 text-green-600",
                              )}
                              onClick={(e) => {
                                e.stopPropagation();
                                void handleVerdictEvent(
                                  event.id,
                                  event.verdict === "true_positive"
                                    ? null
                                    : "true_positive",
                                  null,
                                );
                              }}
                              title="Mark as true positive"
                            >
                              <ThumbsUp className="h-3.5 w-3.5" />
                            </Button>
                            <Button
                              variant="ghost"
                              size="icon"
                              className={cn(
                                "h-7 w-7 text-muted-foreground hover:bg-red-500/10 hover:text-red-600",
                                event.verdict === "false_positive" &&
                                  "bg-red-500/15 text-red-600",
                              )}
                              onClick={(e) => {
                                e.stopPropagation();
                                void handleVerdictEvent(
                                  event.id,
                                  event.verdict === "false_positive"
                                    ? null
                                    : "false_positive",
                                  null,
                                );
                              }}
                              title="Mark as false positive"
                            >
                              <ThumbsDown className="h-3.5 w-3.5" />
                            </Button>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-7 w-7 text-muted-foreground hover:text-destructive"
                              onClick={(e) => handleDeleteEvent(e, event.id)}
                              title="Delete event"
                            >
                              <Trash2 className="h-3.5 w-3.5" />
                            </Button>
                          </div>
                        </div>
                      </CardHeader>
                      <CardContent className="p-3 pt-0">
                        <CardDescription className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
                          <span className="flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            {formatTimestamp(event.timestamp)}
                          </span>
                          {event.confidence != null && (
                            <span>
                              {Math.round(event.confidence * 100)}%
                            </span>
                          )}
                          <span className="flex items-center gap-1">
                            <ImageIcon className="h-3 w-3" />
                            {event.frame_count} frames
                          </span>
                        </CardDescription>
                      </CardContent>
                    </div>

                    {/* Single thumbnail on narrow viewports. The slider in
                        the page header controls its width. */}
                    {thumbFilename && (
                      <div
                        className="shrink-0 self-start overflow-hidden rounded-r-lg border-l bg-muted lg:hidden"
                        style={{
                          width: `${thumbWidth}px`,
                          height: `${filmStripHeight}px`,
                        }}
                      >
                        <ThumbnailImage
                          eventId={event.id}
                          filename={thumbFilename}
                        />
                      </div>
                    )}

                    {/* Filmstrip on wide viewports — keyframes shown side by
                        side so the user can see the gesture across the
                        event without opening the detail dialog. */}
                    {keyframes.length > 0 && (
                      <div
                        className="hidden shrink-0 overflow-hidden rounded-r-lg border-l bg-muted lg:flex"
                        style={{ height: `${filmStripHeight}px` }}
                      >
                        {keyframes.map((filename, i) => (
                          <div
                            key={filename}
                            className={cn(
                              "h-full overflow-hidden",
                              i > 0 && "border-l border-border/60",
                            )}
                            style={{ width: `${thumbWidth}px` }}
                          >
                            <ThumbnailImage
                              eventId={event.id}
                              filename={filename}
                            />
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </Card>
              );
            })}
          </div>
        </ScrollArea>
      )}

      {/* Detail dialog */}
      <EventDetailView
        eventId={selectedEvent}
        open={detailOpen}
        onClose={() => {
          setDetailOpen(false);
          setSelectedEvent(null);
        }}
        onRate={handleRateEvent}
        onVerdict={handleVerdictEvent}
      />
    </div>
  );
}
