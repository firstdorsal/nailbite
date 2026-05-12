import { useState, useEffect, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  Activity,
  AlertTriangle,
  Check,
  Download,
  Lightbulb,
  RefreshCw,
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
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { useConfig } from "@/hooks/useConfig";
import { cn } from "@/lib/utils";
import type { BfrbType, NailbiteConfig } from "@/types";

interface VerdictCounts {
  true_positive: number;
  false_positive: number;
  unsure: number;
}

interface ThresholdSuggestion {
  parameter: string;
  current_value: number;
  proposed_value: number;
  tp_kept: number;
  tp_lost: number;
  fp_kept: number;
  fp_killed: number;
  precision: number;
  recall: number;
}

interface BfrbAnalysis {
  bfrb_type: string;
  counts: VerdictCounts;
  current_confidence_threshold: number;
  confidence_suggestions: ThresholdSuggestion[];
  current_proximity_threshold: number;
  proximity_suggestions: ThresholdSuggestion[];
  events_without_explanation: number;
}

interface LabelAnalysis {
  total_labeled: number;
  per_bfrb: BfrbAnalysis[];
}

interface DatasetExportResult {
  target_dir: string;
  events_exported: number;
  frames_copied: number;
  manifest_path: string;
}

function formatBfrbType(bfrbType: string): string {
  return bfrbType
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

function isInteresting(s: ThresholdSuggestion, current: number): boolean {
  // A suggestion is "interesting" only if it changes the trade-off vs current.
  // Skip rows that are exactly the current threshold (rendered separately).
  if (Math.abs(s.proposed_value - current) < 1e-3) return false;
  // Skip rows that strictly worsen both metrics (lower precision AND recall).
  return true;
}

/**
 * Pick a few "Pareto-helpful" suggestions: better precision than current
 * with non-trivial sample, better recall than current, or extreme settings.
 */
function pickHighlights(
  suggestions: ThresholdSuggestion[],
  current: number,
): ThresholdSuggestion[] {
  const currentRow = suggestions.find(
    (s) => Math.abs(s.proposed_value - current) < 1e-3,
  );
  const currentPrecision = currentRow?.precision ?? 0;
  const currentRecall = currentRow?.recall ?? 0;

  const out: ThresholdSuggestion[] = [];
  // Best precision ≥ 0.5 recall (precision boost without giving up much).
  const bestPrecisionWithRecall = [...suggestions]
    .filter((s) => isInteresting(s, current) && s.recall >= 0.5)
    .sort((a, b) => b.precision - a.precision || b.recall - a.recall)[0];
  if (
    bestPrecisionWithRecall &&
    bestPrecisionWithRecall.precision > currentPrecision + 0.01
  ) {
    out.push(bestPrecisionWithRecall);
  }

  // Best recall (lower threshold, catch more).
  const bestRecall = [...suggestions]
    .filter((s) => isInteresting(s, current))
    .sort((a, b) => b.recall - a.recall || b.precision - a.precision)[0];
  if (
    bestRecall &&
    bestRecall.recall > currentRecall + 0.01 &&
    !out.includes(bestRecall)
  ) {
    out.push(bestRecall);
  }

  // F1-style balance.
  const f1 = (s: ThresholdSuggestion) =>
    s.precision + s.recall > 0
      ? (2 * s.precision * s.recall) / (s.precision + s.recall)
      : 0;
  const bestF1 = [...suggestions]
    .filter((s) => isInteresting(s, current))
    .sort((a, b) => f1(b) - f1(a))[0];
  if (bestF1 && !out.includes(bestF1)) {
    out.push(bestF1);
  }

  return out;
}

function applyThresholdToConfig(
  config: NailbiteConfig,
  bfrb: string,
  parameter: "confidence_threshold" | "proximity_threshold",
  next: number,
): NailbiteConfig {
  const updated = JSON.parse(JSON.stringify(config)) as NailbiteConfig;
  const key = bfrb as BfrbType;
  const behavior = updated.detection.behaviors[key];
  if (!behavior) return updated;
  if (parameter === "confidence_threshold") {
    behavior.confidence_threshold = next;
  } else {
    behavior.proximity_threshold = next;
  }
  return updated;
}

interface SuggestionRowProps {
  suggestion: ThresholdSuggestion;
  isCurrent: boolean;
  onApply?: (proposedValue: number) => void;
  applyBusy: boolean;
}

function SuggestionRow({
  suggestion: s,
  isCurrent,
  onApply,
  applyBusy,
}: SuggestionRowProps) {
  return (
    <div
      className={cn(
        "grid grid-cols-[6rem_1fr_1fr_1fr_auto] items-center gap-3 rounded border px-3 py-2 text-xs",
        isCurrent ? "border-primary/40 bg-primary/5" : "border-border bg-card",
      )}
    >
      <div className="flex items-center gap-1.5">
        <span className="font-mono text-sm font-medium tabular-nums">
          {s.proposed_value.toFixed(2)}
        </span>
        {isCurrent && <Badge variant="secondary">current</Badge>}
      </div>
      <div>
        <div className="text-muted-foreground">precision</div>
        <div className="tabular-nums">{(s.precision * 100).toFixed(0)}%</div>
      </div>
      <div>
        <div className="text-muted-foreground">recall</div>
        <div className="tabular-nums">{(s.recall * 100).toFixed(0)}%</div>
      </div>
      <div>
        <div className="text-muted-foreground">labels</div>
        <div className="tabular-nums">
          <span className="text-green-600">+{s.tp_kept}</span>
          {" / "}
          <span className="text-red-600">+{s.fp_kept}</span>
          <span className="text-muted-foreground">
            {" "}({s.fp_killed} FP killed, {s.tp_lost} TP lost)
          </span>
        </div>
      </div>
      <div>
        {!isCurrent && onApply && (
          <Button
            size="sm"
            variant="outline"
            disabled={applyBusy}
            onClick={() => onApply(s.proposed_value)}
          >
            Apply
          </Button>
        )}
      </div>
    </div>
  );
}

interface ProximitySectionProps {
  bfrb: BfrbAnalysis;
  applyingFor: string | null;
  onApply: (proposedValue: number) => void;
}

function ProximitySection({
  bfrb,
  applyingFor,
  onApply,
}: ProximitySectionProps) {
  const current = bfrb.current_proximity_threshold;
  const sweep = bfrb.proximity_suggestions;
  const usable = sweep.length > 0;
  const currentRow = sweep.find(
    (s) => Math.abs(s.proposed_value - current) < 1e-3,
  );
  const highlights = pickHighlights(sweep, current);

  if (!usable) {
    return (
      <div className="space-y-2">
        <div className="text-xs font-medium text-muted-foreground">
          Proximity threshold
        </div>
        <p className="text-xs text-muted-foreground">
          {bfrb.events_without_explanation > 0
            ? `Older events (${bfrb.events_without_explanation}) lack signal data. Label new events to unlock proximity tuning.`
            : "No proximity suggestions yet."}
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="text-xs font-medium text-muted-foreground">
        Proximity threshold
      </div>
      {currentRow && (
        <SuggestionRow
          suggestion={currentRow}
          isCurrent
          applyBusy={false}
        />
      )}
      {highlights.length === 0 ? (
        <p className="text-xs text-muted-foreground">
          No proximity change improves over the current setting on the
          labeled data.
        </p>
      ) : (
        <>
          <div className="mt-1 flex items-center gap-2 text-xs">
            <Activity className="h-3.5 w-3.5" />
            Suggested alternatives
          </div>
          {highlights.map((s) => (
            <SuggestionRow
              key={s.proposed_value}
              suggestion={s}
              isCurrent={false}
              onApply={onApply}
              applyBusy={
                applyingFor ===
                `${bfrb.bfrb_type}:proximity_threshold:${s.proposed_value}`
              }
            />
          ))}
        </>
      )}
      {bfrb.events_without_explanation > 0 && (
        <p className="text-[10px] text-muted-foreground">
          {bfrb.events_without_explanation} older event(s) excluded — recorded
          before signal capture was available.
        </p>
      )}
    </div>
  );
}

export default function InsightsPage() {
  const { config, saveConfig } = useConfig();
  const [analysis, setAnalysis] = useState<LabelAnalysis | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [applyingFor, setApplyingFor] = useState<string | null>(null);
  const [exportResult, setExportResult] = useState<
    DatasetExportResult | string | null
  >(null);
  const [exporting, setExporting] = useState(false);
  const [exportTarget, setExportTarget] = useState("~/nailbite-dataset");

  const reload = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const result = await invoke<LabelAnalysis>("analyze_labels");
      setAnalysis(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  const handleApply = async (
    bfrb: string,
    parameter: "confidence_threshold" | "proximity_threshold",
    nextValue: number,
  ) => {
    if (!config) return;
    setApplyingFor(`${bfrb}:${parameter}:${nextValue}`);
    try {
      const updated = applyThresholdToConfig(config, bfrb, parameter, nextValue);
      await saveConfig(updated);
      await reload();
    } catch (e) {
      console.error("Failed to apply threshold:", e);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setApplyingFor(null);
    }
  };

  const handleExport = async () => {
    if (!exportTarget.trim()) return;
    setExportResult(null);
    setError(null);
    setExporting(true);
    try {
      const result = await invoke<DatasetExportResult>(
        "export_labeled_dataset",
        { targetDir: exportTarget.trim() },
      );
      setExportResult(result);
    } catch (e) {
      setExportResult(
        `Export failed: ${e instanceof Error ? e.message : String(e)}`,
      );
    } finally {
      setExporting(false);
    }
  };

  if (loading) {
    return (
      <div className="flex h-full flex-col gap-4">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-32 w-full" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-4">
        <AlertTriangle className="h-12 w-12 text-destructive" />
        <p className="text-destructive">{error}</p>
        <Button onClick={reload} variant="outline">
          Retry
        </Button>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col gap-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Insights</h1>
          <p className="text-sm text-muted-foreground">
            {analysis?.total_labeled ?? 0} labeled events analyzed
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button onClick={reload} variant="outline" size="sm">
            <RefreshCw className="mr-2 h-4 w-4" />
            Refresh
          </Button>
          <Input
            value={exportTarget}
            onChange={(e) => setExportTarget(e.target.value)}
            className="h-9 w-56 text-xs"
            placeholder="Target dir (e.g. ~/nailbite-dataset)"
          />
          <Button
            onClick={handleExport}
            variant="outline"
            size="sm"
            disabled={exporting || !exportTarget.trim()}
          >
            <Download className="mr-2 h-4 w-4" />
            {exporting ? "Exporting…" : "Export dataset"}
          </Button>
        </div>
      </div>

      {exportResult && (
        <Card className="border-primary/30 bg-primary/5">
          <CardContent className="pt-4 text-sm">
            {typeof exportResult === "string" ? (
              <span className="text-destructive">{exportResult}</span>
            ) : (
              <span>
                Exported {exportResult.events_exported} events (
                {exportResult.frames_copied} frames) →{" "}
                <span className="font-mono text-xs">
                  {exportResult.manifest_path}
                </span>
              </span>
            )}
          </CardContent>
        </Card>
      )}

      <Separator />

      {!analysis || analysis.per_bfrb.length === 0 ? (
        <div className="flex flex-1 flex-col items-center justify-center gap-3 text-muted-foreground">
          <Lightbulb className="h-12 w-12" />
          <p>No labeled events yet.</p>
          <p className="text-xs">
            Mark detections as true/false positive to build a tuning baseline.
          </p>
        </div>
      ) : (
        <div className="grid gap-4">
          {analysis.per_bfrb.map((b) => {
            const highlights = pickHighlights(
              b.confidence_suggestions,
              b.current_confidence_threshold,
            );
            const currentRow = b.confidence_suggestions.find(
              (s) =>
                Math.abs(
                  s.proposed_value - b.current_confidence_threshold,
                ) < 1e-3,
            );
            const labeledTotal =
              b.counts.true_positive + b.counts.false_positive;
            return (
              <Card key={b.bfrb_type}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle>{formatBfrbType(b.bfrb_type)}</CardTitle>
                    <div className="flex items-center gap-2">
                      <Badge
                        variant="outline"
                        className="border-green-500/40 bg-green-500/10 text-green-600"
                      >
                        <Check className="mr-1 h-3 w-3" />
                        {b.counts.true_positive} TP
                      </Badge>
                      <Badge
                        variant="outline"
                        className="border-red-500/40 bg-red-500/10 text-red-600"
                      >
                        {b.counts.false_positive} FP
                      </Badge>
                      {b.counts.unsure > 0 && (
                        <Badge variant="outline">
                          {b.counts.unsure} unsure
                        </Badge>
                      )}
                    </div>
                  </div>
                  <CardDescription>
                    Current confidence threshold:{" "}
                    <span className="font-mono">
                      {b.current_confidence_threshold.toFixed(2)}
                    </span>
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-2">
                  {labeledTotal === 0 ? (
                    <p className="text-sm text-muted-foreground">
                      No TP/FP labels for this BFRB yet — label some events to
                      unlock threshold suggestions.
                    </p>
                  ) : (
                    <div className="space-y-3">
                      <div className="space-y-2">
                        <div className="text-xs font-medium text-muted-foreground">
                          Confidence threshold
                        </div>
                        {currentRow && (
                          <SuggestionRow
                            suggestion={currentRow}
                            isCurrent
                            applyBusy={false}
                          />
                        )}
                        {highlights.length === 0 ? (
                          <p className="text-xs text-muted-foreground">
                            No confidence change improves over the current
                            setting on the labeled data.
                          </p>
                        ) : (
                          <>
                            <div className="mt-1 flex items-center gap-2 text-xs">
                              <Activity className="h-3.5 w-3.5" />
                              Suggested alternatives
                            </div>
                            {highlights.map((s) => (
                              <SuggestionRow
                                key={s.proposed_value}
                                suggestion={s}
                                isCurrent={false}
                                onApply={(v) =>
                                  handleApply(
                                    b.bfrb_type,
                                    "confidence_threshold",
                                    v,
                                  )
                                }
                                applyBusy={
                                  applyingFor ===
                                  `${b.bfrb_type}:confidence_threshold:${s.proposed_value}`
                                }
                              />
                            ))}
                          </>
                        )}
                      </div>

                      <ProximitySection
                        bfrb={b}
                        applyingFor={applyingFor}
                        onApply={(v) =>
                          handleApply(b.bfrb_type, "proximity_threshold", v)
                        }
                      />
                    </div>
                  )}
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
