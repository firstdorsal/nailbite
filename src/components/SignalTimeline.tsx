import type { DetectionExplanation, FrameInfo } from "@/types";
import { cn } from "@/lib/utils";

interface SignalTimelineProps {
  frames: FrameInfo[];
  /** Index of the currently-shown frame, used to draw a marker. */
  activeIndex: number;
  onSelect?: (index: number) => void;
  className?: string;
}

const DETECTOR_COLORS: Record<string, string> = {
  nail_biting: "stroke-red-500",
  nail_picking: "stroke-amber-500",
  hair_pulling: "stroke-purple-500",
  skin_picking: "stroke-emerald-500",
  lip_biting: "stroke-pink-500",
};

/**
 * Picks the strongest (smallest normalized_distance / threshold) signal from
 * a detector's explanation on a single frame. Returns ratio and confidence.
 */
function pickStrongest(exp: DetectionExplanation): {
  ratio: number;
  confidence: number;
} | null {
  if (exp.hands.length === 0) {
    return null;
  }
  let best = { ratio: Number.POSITIVE_INFINITY, confidence: 0 };
  for (const h of exp.hands) {
    const r = h.distance_threshold > 0
      ? h.normalized_distance / h.distance_threshold
      : Number.POSITIVE_INFINITY;
    if (r < best.ratio) {
      best = { ratio: r, confidence: h.confidence };
    }
  }
  return Number.isFinite(best.ratio) ? best : null;
}

/**
 * Per-frame signal timeline. Each detector gets its own polyline showing
 * `normalized_distance / threshold` across the event's frames, where:
 *  - 0.0 = at the target (strongest signal)
 *  - 1.0 = at threshold (the dashed line)
 *  - >1.0 = above threshold (no contribution)
 *
 * The line is inverted on the y-axis so "stronger signal = higher line",
 * which is the intuitive direction for users reading the chart.
 */
export function SignalTimeline({
  frames,
  activeIndex,
  onSelect,
  className,
}: SignalTimelineProps) {
  if (frames.length === 0) {
    return null;
  }

  const width = 100;
  const height = 32;
  const padding = 1;

  // Collect detector ids that appear in any frame.
  const detectors = new Set<string>();
  for (const f of frames) {
    for (const e of f.explanations) {
      detectors.add(e.bfrb_type);
    }
  }
  const detectorList = Array.from(detectors);

  if (detectorList.length === 0) {
    return (
      <div className={cn("text-xs text-muted-foreground", className)}>
        No per-frame signals recorded for this event.
      </div>
    );
  }

  // y-mapping: ratio 0 => height-padding (bottom-up display) ... ratio 1 (threshold) => height/2
  // We clip the visible range to [0, 1.5].
  const maxRatio = 1.5;
  const yFor = (ratio: number) => {
    const clamped = Math.max(0, Math.min(maxRatio, ratio));
    const fillFromZero = 1 - clamped / maxRatio; // 1 at ratio=0, 0 at ratio=maxRatio
    return padding + (1 - fillFromZero) * (height - padding * 2);
  };
  const thresholdY = yFor(1.0);

  // Build polyline for each detector. Use null gaps when there is no data.
  const stepX = (width - padding * 2) / Math.max(frames.length - 1, 1);

  return (
    <div className={cn("space-y-1", className)}>
      <div className="relative">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          preserveAspectRatio="none"
          className="h-12 w-full rounded border bg-card"
          aria-label="Signal proximity over time"
        >
          {/* Threshold line */}
          <line
            x1={padding}
            x2={width - padding}
            y1={thresholdY}
            y2={thresholdY}
            className="stroke-foreground/30"
            strokeWidth={0.3}
            strokeDasharray="1 1"
          />

          {/* Detector lines */}
          {detectorList.map((id) => {
            const points = frames
              .map((f, i) => {
                const exp = f.explanations.find((e) => e.bfrb_type === id);
                if (!exp) return null;
                const best = pickStrongest(exp);
                if (best === null) return null;
                const x = padding + i * stepX;
                const y = yFor(best.ratio);
                return `${x.toFixed(2)},${y.toFixed(2)}`;
              })
              .filter((p): p is string => p !== null)
              .join(" ");
            return (
              <polyline
                key={id}
                points={points}
                className={cn(
                  "fill-none",
                  DETECTOR_COLORS[id] ?? "stroke-blue-500",
                )}
                strokeWidth={0.6}
                vectorEffect="non-scaling-stroke"
              />
            );
          })}

          {/* Active frame marker */}
          {activeIndex >= 0 && activeIndex < frames.length && (
            <line
              x1={padding + activeIndex * stepX}
              x2={padding + activeIndex * stepX}
              y1={padding}
              y2={height - padding}
              className="stroke-primary"
              strokeWidth={0.4}
            />
          )}

          {/* Trigger marker (offset 0) */}
          {(() => {
            const triggerIdx = frames.findIndex((f) => f.offset === 0);
            if (triggerIdx < 0) return null;
            const x = padding + triggerIdx * stepX;
            return (
              <line
                x1={x}
                x2={x}
                y1={padding}
                y2={height - padding}
                className="stroke-destructive/60"
                strokeWidth={0.3}
                strokeDasharray="1 1"
              />
            );
          })()}

          {/* Click handler band per frame */}
          {onSelect &&
            frames.map((_, i) => (
              <rect
                key={i}
                x={padding + (i - 0.5) * stepX}
                y={0}
                width={stepX}
                height={height}
                className="cursor-pointer fill-transparent"
                onClick={() => onSelect(i)}
              />
            ))}
        </svg>
      </div>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[10px] text-muted-foreground">
        {detectorList.map((id) => (
          <span key={id} className="flex items-center gap-1">
            <span
              className={cn(
                "inline-block h-0.5 w-3",
                (DETECTOR_COLORS[id] ?? "stroke-blue-500").replace("stroke-", "bg-"),
              )}
            />
            {id.replace(/_/g, " ")}
          </span>
        ))}
        <span className="ml-auto">
          <span className="inline-block border-t border-dashed border-foreground/30 px-1.5 align-middle" />{" "}
          threshold
        </span>
      </div>
    </div>
  );
}
