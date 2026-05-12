import type {
  DetectionExplanation,
  HandSignal,
  SuppressionReason,
} from "@/types";
import { cn } from "@/lib/utils";

const SUPPRESSION_LABELS: Record<SuppressionReason, string> = {
  no_face: "no face visible",
  no_hands: "no hands visible",
  typing_posture: "typing posture",
  chin_rest: "chin rest",
  insufficient_hands: "needs 2 hands",
};

const BFRB_LABELS: Record<string, string> = {
  nail_biting: "Nail biting",
  nail_picking: "Nail picking",
  hair_pulling: "Hair pulling",
  skin_picking: "Skin picking",
  lip_biting: "Lip biting",
};

const FINGERTIP_NAMES: Record<number, string> = {
  4: "thumb",
  8: "index",
  12: "middle",
  16: "ring",
  20: "pinky",
};

function formatFingertip(idx: number | null): string {
  if (idx === null) return "—";
  return FINGERTIP_NAMES[idx] ?? `lm${idx}`;
}

function formatSide(side: "Left" | "Right" | null): string {
  if (side === "Left") return "L";
  if (side === "Right") return "R";
  return "?";
}

interface DistanceBarProps {
  /** Distance value (smaller = stronger signal). */
  distance: number;
  /** Threshold above which the signal is rejected. */
  threshold: number;
}

/** Visualizes proximity as a bar: full when distance==0, empty when >= threshold. */
function DistanceBar({ distance, threshold }: DistanceBarProps) {
  // Display range: [0, 1.5 * threshold] so values just over threshold are still visible.
  const max = threshold * 1.5;
  const clamped = Math.min(distance, max);
  const fillFromZero = Math.max(0, 1 - clamped / threshold);
  const overThreshold = distance >= threshold;
  const thresholdPct = (threshold / max) * 100;
  return (
    <div className="relative h-1.5 w-full overflow-hidden rounded bg-muted">
      <div
        className={cn(
          "absolute inset-y-0 left-0 rounded-l",
          overThreshold ? "bg-muted-foreground/40" : "bg-primary",
        )}
        style={{ width: `${fillFromZero * 100}%` }}
      />
      <div
        className="absolute inset-y-0 w-px bg-foreground/30"
        style={{ left: `${thresholdPct}%` }}
        aria-hidden
      />
    </div>
  );
}

interface HandSignalRowProps {
  signal: HandSignal;
}

function HandSignalRow({ signal }: HandSignalRowProps) {
  const ratio = signal.normalized_distance / signal.distance_threshold;
  const fired = signal.confidence > 0;
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between gap-2 text-xs">
        <span className="flex items-baseline gap-1.5">
          <span
            className={cn(
              "rounded px-1 font-mono text-[10px]",
              fired
                ? "bg-primary/20 text-primary"
                : "bg-muted text-muted-foreground",
            )}
          >
            {formatSide(signal.side)}
          </span>
          <span className="text-muted-foreground">
            {formatFingertip(signal.contributing_fingertip)}
            {signal.partner_fingertip !== null && (
              <>
                {" → "}
                {formatFingertip(signal.partner_fingertip)}
              </>
            )}
          </span>
        </span>
        <span
          className={cn(
            "tabular-nums",
            fired ? "font-medium text-foreground" : "text-muted-foreground",
          )}
        >
          {signal.normalized_distance.toFixed(2)}
          <span className="text-muted-foreground"> / {signal.distance_threshold.toFixed(2)}</span>
        </span>
      </div>
      <DistanceBar
        distance={signal.normalized_distance}
        threshold={signal.distance_threshold}
      />
      {(signal.curl !== null || signal.bonus > 0) && (
        <div className="flex justify-end gap-2 text-[10px] text-muted-foreground tabular-nums">
          {signal.curl !== null && <span>curl {signal.curl.toFixed(2)}</span>}
          {signal.bonus > 0 && <span>+{signal.bonus.toFixed(2)} pinch</span>}
          <span>
            ratio {ratio.toFixed(2)}
          </span>
        </div>
      )}
    </div>
  );
}

interface DetectorBlockProps {
  explanation: DetectionExplanation;
  /** Compact = no border/padding wrapper, used inside a parent panel. */
  compact?: boolean;
}

function DetectorBlock({ explanation, compact }: DetectorBlockProps) {
  const label = BFRB_LABELS[explanation.bfrb_type] ?? explanation.bfrb_type;
  const hasHands = explanation.hands.length > 0;
  const hasSuppressions = explanation.suppressions.length > 0;
  return (
    <div className={cn(compact ? "space-y-2" : "space-y-2 rounded-lg border bg-card p-3")}>
      <div className="flex items-baseline justify-between">
        <span className="font-medium">{label}</span>
        <span
          className={cn(
            "text-xs tabular-nums",
            explanation.frame_confidence > 0
              ? "text-foreground"
              : "text-muted-foreground",
          )}
        >
          {(explanation.frame_confidence * 100).toFixed(0)}%
        </span>
      </div>
      {hasSuppressions && (
        <div className="flex flex-wrap gap-1">
          {explanation.suppressions.map((s) => (
            <span
              key={s}
              className="rounded bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground"
            >
              {SUPPRESSION_LABELS[s] ?? s}
            </span>
          ))}
        </div>
      )}
      {hasHands ? (
        <div className="space-y-2">
          {explanation.hands.map((h, i) => (
            <HandSignalRow key={`${h.hand_index}-${i}`} signal={h} />
          ))}
        </div>
      ) : (
        !hasSuppressions && (
          <div className="text-xs text-muted-foreground">no contributing hands</div>
        )
      )}
    </div>
  );
}

export interface SignalsPanelProps {
  /** Per-detector explanations to render. */
  explanations: DetectionExplanation[];
  /** Title shown at the top of the panel; omit for inline use. */
  title?: string;
  /** Render without a card wrapper (for embedding in AlertModal). */
  inline?: boolean;
  /** Class names for the outer container. */
  className?: string;
}

/** Renders the contributing signals for one or more detectors. */
export function SignalsPanel({
  explanations,
  title,
  inline,
  className,
}: SignalsPanelProps) {
  const hasContent = explanations.length > 0;
  const content = (
    <div className="space-y-3">
      {title && <h3 className="font-semibold">{title}</h3>}
      {hasContent ? (
        explanations.map((exp) => (
          <DetectorBlock
            key={exp.bfrb_type}
            explanation={exp}
            compact={inline}
          />
        ))
      ) : (
        <div className="text-xs text-muted-foreground">
          Waiting for detection signals…
        </div>
      )}
    </div>
  );

  if (inline) {
    return <div className={className}>{content}</div>;
  }
  return (
    <div className={cn("rounded-lg border bg-card p-4", className)}>
      {content}
    </div>
  );
}
