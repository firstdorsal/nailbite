import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { cn } from "@/lib/utils";
import {
  RefreshCw,
  AlertTriangle,
  Activity,
  CheckCircle,
  XCircle,
  AlertCircle,
} from "lucide-react";
import type { SessionStats, SessionEntry } from "@/types";

export default function StatsPage() {
  const [stats, setStats] = useState<SessionStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadStats = async () => {
    try {
      setLoading(true);
      setError(null);
      const result = await invoke<SessionStats>("get_stats");
      setStats(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load stats");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadStats();
  }, []);

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  const getEventIcon = (eventType: SessionEntry["event_type"]) => {
    switch (eventType) {
      case "detection":
        return <Activity className="h-4 w-4 text-red-500" />;
      case "exercise_completed":
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case "dismissed":
        return <XCircle className="h-4 w-4 text-yellow-500" />;
      case "missed":
        return <AlertCircle className="h-4 w-4 text-orange-500" />;
    }
  };

  const getEventLabel = (entry: SessionEntry) => {
    switch (entry.event_type) {
      case "detection":
        return `BFRB Detected: ${entry.bfrb_type?.replace("_", " ")}`;
      case "exercise_completed":
        return `Exercise Completed: ${entry.exercise_id}`;
      case "dismissed":
        return "Alert Dismissed";
      case "missed":
        return "Missed Event Marked";
    }
  };

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-4">
        <AlertTriangle className="h-12 w-12 text-destructive" />
        <p className="text-destructive">{error}</p>
        <button
          onClick={loadStats}
          className="rounded-lg bg-primary px-4 py-2 text-primary-foreground"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Statistics</h1>
        <button
          onClick={loadStats}
          className="flex items-center gap-2 rounded-lg bg-secondary px-3 py-2 text-sm"
          disabled={loading}
        >
          <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} />
          Refresh
        </button>
      </div>

      {/* Summary cards */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-lg border bg-card p-4">
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-red-500" />
            <span className="text-sm font-medium text-muted-foreground">
              Detections
            </span>
          </div>
          <p className="mt-2 text-3xl font-bold">
            {stats?.total_detections || 0}
          </p>
        </div>

        <div className="rounded-lg border bg-card p-4">
          <div className="flex items-center gap-2">
            <CheckCircle className="h-5 w-5 text-green-500" />
            <span className="text-sm font-medium text-muted-foreground">
              Exercises Done
            </span>
          </div>
          <p className="mt-2 text-3xl font-bold">
            {stats?.total_exercises_completed || 0}
          </p>
        </div>

        <div className="rounded-lg border bg-card p-4">
          <div className="flex items-center gap-2">
            <XCircle className="h-5 w-5 text-yellow-500" />
            <span className="text-sm font-medium text-muted-foreground">
              Dismissed
            </span>
          </div>
          <p className="mt-2 text-3xl font-bold">
            {stats?.total_dismissed || 0}
          </p>
        </div>

        <div className="rounded-lg border bg-card p-4">
          <div className="flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-orange-500" />
            <span className="text-sm font-medium text-muted-foreground">
              Missed Events
            </span>
          </div>
          <p className="mt-2 text-3xl font-bold">{stats?.total_missed || 0}</p>
        </div>
      </div>

      {/* History table */}
      <div className="flex-1 overflow-hidden rounded-lg border bg-card">
        <div className="border-b bg-muted/50 p-4">
          <h2 className="font-semibold">Event History</h2>
        </div>

        <div className="h-full overflow-auto">
          {stats?.entries && stats.entries.length > 0 ? (
            <table className="w-full">
              <thead className="sticky top-0 bg-card">
                <tr className="border-b text-left text-sm text-muted-foreground">
                  <th className="p-3">Time</th>
                  <th className="p-3">Event</th>
                  <th className="p-3">Details</th>
                </tr>
              </thead>
              <tbody>
                {stats.entries
                  .slice()
                  .reverse()
                  .map((entry, i) => (
                    <tr
                      key={i}
                      className="border-b last:border-b-0 hover:bg-muted/50"
                    >
                      <td className="p-3 text-sm text-muted-foreground">
                        {formatTimestamp(entry.timestamp)}
                      </td>
                      <td className="p-3">
                        <div className="flex items-center gap-2">
                          {getEventIcon(entry.event_type)}
                          <span className="text-sm">
                            {getEventLabel(entry)}
                          </span>
                        </div>
                      </td>
                      <td className="p-3 text-sm text-muted-foreground">
                        {entry.duration_secs &&
                          `Duration: ${entry.duration_secs}s`}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          ) : (
            <div className="flex h-full items-center justify-center p-8 text-muted-foreground">
              No events recorded yet
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
