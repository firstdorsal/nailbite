import type {
  BfrbType,
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
      <div className="flex items-center justify-between gap-2 text-xs">
        <span className="flex min-w-0 items-center gap-1.5">
          <span
            className={cn(
              "shrink-0 rounded px-1 font-mono text-[10px]",
              fired
                ? "bg-primary/20 text-primary"
                : "bg-muted text-muted-foreground",
            )}
          >
            {formatSide(signal.side)}
          </span>
          <span className="truncate text-muted-foreground">
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
            "shrink-0 tabular-nums",
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
      {/* The stats sub-row is always rendered (even when curl/bonus
          are missing) so the row height is constant; missing values
          show an em dash. Without this guarantee, a hand swinging
          between "has curl" and "no curl" would shift everything
          below it by one line. */}
      <div className="flex justify-end gap-2 text-[10px] text-muted-foreground tabular-nums">
        <span>curl {signal.curl !== null ? signal.curl.toFixed(2) : "—"}</span>
        <span>+{signal.bonus.toFixed(2)} pinch</span>
        <span>ratio {ratio.toFixed(2)}</span>
      </div>
    </div>
  );
}

/**
 * One detector's signal block. Layout is height-stable:
 *   - Suppression chips reserve a single line of vertical space whether
 *     or not any chips are visible.
 *   - The hands area always renders exactly two rows so a detector
 *     swinging between 0, 1, and 2 contributing hands doesn't shift
 *     adjacent UI vertically.
 *
 * When the detector has no live explanation (e.g. it's enabled but
 * isn't seeing the relevant body part right now), pass `explanation =
 * null` and the block renders a dimmed placeholder in the same shape.
 */
function DetectorBlock({
  bfrbType,
  explanation,
}: {
  bfrbType: string;
  explanation: DetectionExplanation | null;
}) {
  const label = BFRB_LABELS[bfrbType] ?? bfrbType;
  const hands = explanation?.hands ?? [];
  const suppressions = explanation?.suppressions ?? [];
  const frameConfidence = explanation?.frame_confidence ?? 0;
  return (
    <div
      className={cn(
        "flex flex-1 flex-col gap-2",
        !explanation && "opacity-60",
      )}
    >
      <div className="flex items-baseline justify-between">
        <span className="font-medium">{label}</span>
        <span
          className={cn(
            "text-xs tabular-nums",
            frameConfidence > 0 ? "text-foreground" : "text-muted-foreground",
          )}
        >
          {(frameConfidence * 100).toFixed(0)}%
        </span>
      </div>
      {/* Single-line, fixed-height suppression chip row. Overflow is
          clipped rather than wrapped so two long suppression names
          can never bump the block height. */}
      <div className="flex h-[18px] items-center gap-1 overflow-hidden whitespace-nowrap">
        {suppressions.length > 0 ? (
          suppressions.map((s) => (
            <span
              key={s}
              className="rounded bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground"
            >
              {SUPPRESSION_LABELS[s] ?? s}
            </span>
          ))
        ) : (
          <span className="text-[10px] text-muted-foreground/0">·</span>
        )}
      </div>
      <div className="space-y-2">
        {[0, 1].map((slotIdx) => {
          const h = hands[slotIdx];
          return h ? (
            <HandSignalRow key={`${h.hand_index}-${slotIdx}`} signal={h} />
          ) : (
            <PlaceholderHandRow key={`placeholder-${slotIdx}`} />
          );
        })}
      </div>
    </div>
  );
}

/** Height-matched placeholder for a missing hand slot. Same DOM shape as
 *  HandSignalRow so the block height stays identical regardless of how
 *  many hands are currently contributing. */
function PlaceholderHandRow() {
  return (
    <div className="space-y-1 opacity-40">
      <div className="flex items-baseline justify-between gap-2 text-xs">
        <span className="flex items-baseline gap-1.5">
          <span className="rounded bg-muted px-1 font-mono text-[10px] text-muted-foreground">
            —
          </span>
          <span className="text-muted-foreground">—</span>
        </span>
        <span className="tabular-nums text-muted-foreground">—</span>
      </div>
      <div className="h-1.5 w-full rounded bg-muted" />
      <div className="flex justify-end gap-2 text-[10px] text-muted-foreground tabular-nums">
        <span>curl —</span>
        <span>+0.00 pinch</span>
        <span>ratio —</span>
      </div>
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

/** Detectors we always reserve space for in the panel. Order is fixed
 *  so the row positions never reshuffle when a detector starts/stops
 *  emitting explanations. New detectors get a new fixed slot here. */
const PANEL_DETECTOR_SLOTS: readonly BfrbType[] = ["nail_biting", "nail_picking"];

/** Renders the contributing signals for one or more detectors. The
 *  panel reserves space for every known detector (see
 *  `PANEL_DETECTOR_SLOTS`); detectors with no live explanation show a
 *  dimmed placeholder so the height stays constant across frames.
 *
 *  When used inline (e.g. inside `AlertModal`) we drop the placeholder
 *  rows and render only the detector that fired — the alert modal is
 *  scoped to a single event, not a live monitoring view. */
export function SignalsPanel({
  explanations,
  title,
  inline,
  className,
}: SignalsPanelProps) {
  const slots = inline
    ? explanations.map((e) => e.bfrb_type)
    : PANEL_DETECTOR_SLOTS;
  const byType = new Map(explanations.map((e) => [e.bfrb_type, e]));
  // Inline mode (AlertModal) lays out at natural height. Full mode
  // (Preview panel) stretches to fill its parent so the detector
  // blocks divide the available vertical space evenly — that keeps
  // the panel's bottom edge aligned with the camera beside it.
  const content = (
    <div className={cn(inline ? "space-y-3" : "flex h-full flex-col gap-3")}>
      {title && <h3 className="font-semibold">{title}</h3>}
      {slots.map((bfrbType) => (
        <DetectorBlock
          key={bfrbType}
          bfrbType={bfrbType}
          explanation={byType.get(bfrbType) ?? null}
        />
      ))}
    </div>
  );
  return <div className={cn("h-full", className)}>{content}</div>;
}
