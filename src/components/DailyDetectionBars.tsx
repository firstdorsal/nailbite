import { useMemo } from "react";
import { cn } from "@/lib/utils";

interface DailyDetectionBarsProps {
  /**
   * Events with a parseable ISO timestamp; same shape as EventHistorySummary.
   * `trigger` is optional for back-compat but, when present, is used to
   * exclude `false_positive` and `missed_event` rows so the bar counts
   * line up with the in-app "today" badge (which only counts confirmed
   * detections).
   */
  events: Array<{ timestamp: string; trigger?: string }>;
  /** Currently-selected day key in YYYY-MM-DD (local time), or null = all. */
  selectedDate: string | null;
  /** Called with a day key when the user clicks a bar; or null when "All" is hit. */
  onSelectDate: (date: string | null) => void;
  /** Number of trailing days to render. Defaults to 14. */
  days?: number;
}

/** Convert a Date to its YYYY-MM-DD key in the local timezone. */
function toDayKey(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function dayLabel(d: Date, isToday: boolean): string {
  if (isToday) return "Today";
  // Short weekday + day-of-month, e.g. "Mon 9".
  const wd = d.toLocaleDateString(undefined, { weekday: "short" });
  return `${wd} ${d.getDate()}`;
}

/**
 * Horizontal strip of small bars — one bar per day for the last N days,
 * height proportional to the day's detection count. Click a bar to filter
 * the event list to that day; click the active bar (or "All") to clear.
 */
export function DailyDetectionBars({
  events,
  selectedDate,
  onSelectDate,
  days = 14,
}: DailyDetectionBarsProps) {
  const series = useMemo(() => {
    // Build counts per local-day key. We only count confirmed detection
    // triggers — false-positive dismissals create their own event rows
    // in the history, but doubling the bar height for every dismissed
    // detection would diverge from the in-app "today" badge.
    const counts = new Map<string, number>();
    for (const ev of events) {
      if (ev.trigger && ev.trigger !== "detection") continue;
      const d = new Date(ev.timestamp);
      if (Number.isNaN(d.getTime())) continue;
      const key = toDayKey(d);
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }

    // Generate the last `days` keys ending today.
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const out: Array<{ key: string; date: Date; count: number; isToday: boolean }> = [];
    for (let i = days - 1; i >= 0; i -= 1) {
      const d = new Date(today);
      d.setDate(today.getDate() - i);
      const key = toDayKey(d);
      out.push({
        key,
        date: d,
        count: counts.get(key) ?? 0,
        isToday: i === 0,
      });
    }
    return out;
  }, [events, days]);

  const max = useMemo(
    () => Math.max(1, ...series.map((s) => s.count)),
    [series],
  );

  return (
    <div className="space-y-2 rounded-lg border bg-card p-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-muted-foreground">
          Last {days} days
        </span>
        <button
          type="button"
          onClick={() => onSelectDate(null)}
          className={cn(
            "rounded px-2 py-0.5 text-[11px] transition-colors",
            selectedDate === null
              ? "bg-primary text-primary-foreground"
              : "text-muted-foreground hover:bg-accent",
          )}
        >
          All
        </button>
      </div>

      {/* items-stretch + justify-end inside each button gives every column
          the same row height (= the tallest bar's column), with the bar
          and labels anchored to the bottom. That makes the WHOLE column
          a click target — short bars no longer leave dead zone above
          themselves. Horizontal gap tightened so days read denser. */}
      <div className="flex items-stretch gap-0.5">
        {series.map((s) => {
          const active = selectedDate === s.key;
          // Height: 4px floor for empty days so the row stays visible,
          // up to 56px for the busiest day.
          const heightPx = s.count === 0 ? 4 : 4 + Math.round((s.count / max) * 52);
          return (
            <button
              key={s.key}
              type="button"
              onClick={() => onSelectDate(active ? null : s.key)}
              className={cn(
                "group flex flex-1 flex-col items-center justify-end gap-1 rounded-sm px-0.5 py-1 transition-colors hover:bg-accent/40",
                active && "bg-accent",
              )}
              title={`${s.date.toLocaleDateString(undefined, {
                weekday: "long",
                year: "numeric",
                month: "short",
                day: "numeric",
              })}: ${s.count} detection${s.count === 1 ? "" : "s"}`}
            >
              <span
                className={cn(
                  "block w-full max-w-[24px] rounded-sm transition-colors",
                  s.count === 0
                    ? "bg-muted-foreground/20"
                    : active
                      ? "bg-primary"
                      : "bg-primary/60 group-hover:bg-primary/80",
                )}
                style={{ height: `${heightPx}px` }}
              />
              <span
                className={cn(
                  "text-[10px] tabular-nums",
                  active
                    ? "font-medium text-foreground"
                    : "text-muted-foreground",
                )}
              >
                {dayLabel(s.date, s.isToday)}
              </span>
              <span
                className={cn(
                  "text-[9px] tabular-nums",
                  s.count === 0 ? "text-muted-foreground/40" : "text-muted-foreground",
                )}
              >
                {s.count}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

/** Re-exported so callers can match on the same key format. */
export const dayKeyOf = toDayKey;
