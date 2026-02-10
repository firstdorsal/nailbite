import { cn } from "@/lib/utils";

export type Status = "normal" | "alert" | "paused" | "offline";

interface StatusIndicatorProps {
  status: Status;
  className?: string;
  showLabel?: boolean;
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
};

export function StatusIndicator({
  status,
  className,
  showLabel = true,
}: StatusIndicatorProps) {
  const config = statusConfig[status];

  return (
    <div className={cn("flex items-center gap-2", className)}>
      <span className="relative flex h-3 w-3">
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
            "relative inline-flex h-3 w-3 rounded-full",
            config.color
          )}
        />
      </span>
      {showLabel && (
        <span className="text-sm font-medium text-muted-foreground">
          {config.label}
        </span>
      )}
    </div>
  );
}
