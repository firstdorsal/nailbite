import { cn } from "@/lib/utils";

export type Status = "normal" | "alert" | "paused" | "offline" | "absent";

interface StatusIndicatorProps {
  status: Status;
  className?: string;
  showLabel?: boolean;
  /**
   * Optional detection-count badge rendered to the right of the status dot.
   * When `undefined` or `null`, no badge is shown. Pass `0` to render "0".
   */
  count?: number | null;
}

const statusConfig: Record<
  Status,
  { color: string; label: string; pulse: boolean }
> = {
  normal: {
    color: "bg-green-500",
    label: "Monitoring",
    pulse: true,
  },
  alert: {
    color: "bg-red-500",
    label: "Alert",
    pulse: true,
  },
  paused: {
    color: "bg-yellow-500",
    label: "Paused",
    pulse: false,
  },
  offline: {
    color: "bg-gray-500",
    label: "Offline",
    pulse: false,
  },
  absent: {
    // Mirrors `TrayState::Absent` in src-tauri/src/tray.rs — keep colors in sync.
    color: "bg-gray-700",
    label: "No one in frame",
    pulse: false,
  },
};

export function StatusIndicator({
  status,
  className,
  showLabel = true,
  count,
}: StatusIndicatorProps) {
  const config = statusConfig[status];
  const showCount = typeof count === "number";
  // Slightly larger than the empty dot so the digits actually fit. We
  // keep the dot small (h-3 w-3) when no count is shown so the layout
  // matches the rest of the UI's existing rhythm.
  const dotSize = showCount ? "h-5 w-5" : "h-3 w-3";

  return (
    <div className={cn("flex items-center gap-2", className)}>
      <span className={cn("relative flex items-center justify-center", dotSize)}>
        {config.pulse && (
          <span
            className={cn(
              "absolute inline-flex h-full w-full animate-ping rounded-full opacity-75",
              config.color
            )}
          />
        )}
        <span
          className={cn(
            "relative inline-flex items-center justify-center rounded-full",
            dotSize,
            config.color,
          )}
          title={
            showCount
              ? `${count} detection${count === 1 ? "" : "s"} today`
              : undefined
          }
        >
          {showCount && (
            <span className="text-[10px] font-bold leading-none tabular-nums text-white">
              {count}
            </span>
          )}
        </span>
      </span>
      {showLabel && (
        <span className="text-sm font-medium text-muted-foreground">
          {config.label}
        </span>
      )}
    </div>
  );
}
