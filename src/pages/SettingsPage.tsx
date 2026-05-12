import { useConfig } from "@/hooks/useConfig";
import { cn } from "@/lib/utils";
import { RefreshCw, AlertTriangle } from "lucide-react";
import type { NailbiteConfig } from "@/types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";

export default function SettingsPage() {
  // Settings auto-save: every change is pushed through `patchConfig`, which
  // updates the shared context optimistically and debounces the backend
  // save so slider drags don't trigger one save per frame. There is no
  // explicit Save button — the Settings page is the live source of truth.
  const { config, loading, error, patchConfig, reloadConfig } = useConfig();

  const updateConfig = <K extends keyof NailbiteConfig>(
    section: K,
    updates: Partial<NailbiteConfig[K]>,
  ) => {
    if (!config) return;
    patchConfig({
      ...config,
      [section]: { ...config[section], ...updates },
    });
  };

  // Read directly from the shared context so the page reflects any external
  // changes immediately. `localConfig` was a stale local mirror.
  const localConfig = config;

  if (loading) {
    return (
      <div className="flex h-full flex-col gap-4">
        <div className="flex items-center justify-between">
          <Skeleton className="h-8 w-32" />
          <div className="flex gap-2">
            <Skeleton className="h-10 w-24" />
            <Skeleton className="h-10 w-20" />
          </div>
        </div>
        <div className="space-y-6">
          <Skeleton className="h-48 w-full" />
          <Skeleton className="h-64 w-full" />
          <Skeleton className="h-48 w-full" />
        </div>
      </div>
    );
  }

  if (error || !localConfig) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-4">
        <AlertTriangle className="h-12 w-12 text-destructive" />
        <p className="text-destructive">{error || "Failed to load config"}</p>
        <Button onClick={reloadConfig} variant="outline">
          Retry
        </Button>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Settings</h1>
          <p className="text-xs text-muted-foreground">
            Changes are saved automatically.
          </p>
        </div>
        <Button
          onClick={reloadConfig}
          variant="secondary"
          size="sm"
          disabled={loading}
        >
          <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} />
          Reload
        </Button>
      </div>

      {/* Save error */}
      {error && (
        <div className="flex items-center gap-2 rounded-lg bg-destructive/10 p-4 text-destructive">
          <AlertTriangle className="h-5 w-5 shrink-0" />
          <span className="text-sm">{error}</span>
        </div>
      )}

      {/* Settings sections */}
      <div className="flex-1 space-y-6 overflow-auto pb-4">
        {/* General */}
        <Card>
          <CardHeader className="pb-4">
            <CardTitle className="text-lg">General</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="log-level">Log Level</Label>
              <Select
                value={localConfig.general.log_level}
                onValueChange={(value) =>
                  updateConfig("general", { log_level: value })
                }
              >
                <SelectTrigger id="log-level">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="trace">Trace</SelectItem>
                  <SelectItem value="debug">Debug</SelectItem>
                  <SelectItem value="info">Info</SelectItem>
                  <SelectItem value="warn">Warn</SelectItem>
                  <SelectItem value="error">Error</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="cooldown">Cooldown (seconds)</Label>
              <Input
                id="cooldown"
                type="number"
                value={localConfig.general.cooldown_seconds}
                onChange={(e) =>
                  updateConfig("general", {
                    cooldown_seconds: parseInt(e.target.value) || 0,
                  })
                }
                min={0}
              />
            </div>
            <div className="flex items-center gap-3 sm:col-span-2">
              <Switch
                id="show-detection-count"
                checked={localConfig.general.show_detection_count}
                onCheckedChange={(checked) =>
                  updateConfig("general", {
                    show_detection_count: checked,
                  })
                }
              />
              <Label htmlFor="show-detection-count" className="cursor-pointer">
                Show today&apos;s detection count in the status indicator and
                tray tooltip
              </Label>
            </div>
          </CardContent>
        </Card>

        {/* Camera */}
        <Card>
          <CardHeader className="pb-4">
            <CardTitle className="text-lg">Camera</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="inference-fps">Inference FPS</Label>
              <Input
                id="inference-fps"
                type="number"
                value={localConfig.camera.inference_fps}
                onChange={(e) =>
                  updateConfig("camera", {
                    inference_fps: parseInt(e.target.value) || 8,
                  })
                }
                min={1}
                max={30}
              />
            </div>
            <div className="space-y-3">
              <Label>Camera Sources</Label>
              {localConfig.camera.sources.map((source, index) => (
                <div
                  key={source.id}
                  className="rounded-lg border bg-muted/50 p-3"
                >
                  <div className="mb-2 flex items-center justify-between">
                    <span className="text-sm font-medium">{source.id}</span>
                    <Badge
                      variant={
                        source.role === "primary" ? "default" : "secondary"
                      }
                    >
                      {source.role}
                    </Badge>
                  </div>
                  <div className="grid gap-2 sm:grid-cols-3">
                    <div className="space-y-1">
                      <Label className="text-xs text-muted-foreground">
                        Device
                      </Label>
                      <Input
                        value={source.device}
                        onChange={(e) => {
                          const newSources = [...localConfig.camera.sources];
                          newSources[index] = {
                            ...source,
                            device: e.target.value,
                          };
                          updateConfig("camera", { sources: newSources });
                        }}
                        className="h-8 text-sm"
                      />
                    </div>
                    <div className="space-y-1">
                      <Label className="text-xs text-muted-foreground">
                        Width
                      </Label>
                      <Input
                        type="number"
                        value={source.resolution_width}
                        onChange={(e) => {
                          const newSources = [...localConfig.camera.sources];
                          newSources[index] = {
                            ...source,
                            resolution_width:
                              parseInt(e.target.value) || 640,
                          };
                          updateConfig("camera", { sources: newSources });
                        }}
                        className="h-8 text-sm"
                      />
                    </div>
                    <div className="space-y-1">
                      <Label className="text-xs text-muted-foreground">
                        Height
                      </Label>
                      <Input
                        type="number"
                        value={source.resolution_height}
                        onChange={(e) => {
                          const newSources = [...localConfig.camera.sources];
                          newSources[index] = {
                            ...source,
                            resolution_height:
                              parseInt(e.target.value) || 480,
                          };
                          updateConfig("camera", { sources: newSources });
                        }}
                        className="h-8 text-sm"
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Detection - Behaviors */}
        <Card>
          <CardHeader className="pb-4">
            <CardTitle className="text-lg">Detection Behaviors</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {(
              [
                "nail_biting",
                "nail_picking",
                "hair_pulling",
                "skin_picking",
                "lip_biting",
              ] as const
            ).map((behavior) => (
              <div
                key={behavior}
                className="flex items-center justify-between rounded-lg border bg-muted/50 p-3"
              >
                <div className="flex items-center gap-3">
                  <Switch
                    id={behavior}
                    checked={
                      localConfig.detection.behaviors[behavior].enabled
                    }
                    onCheckedChange={(checked) =>
                      patchConfig({
                        ...localConfig,
                        detection: {
                          ...localConfig.detection,
                          behaviors: {
                            ...localConfig.detection.behaviors,
                            [behavior]: {
                              ...localConfig.detection.behaviors[behavior],
                              enabled: checked,
                            },
                          },
                        },
                      })
                    }
                  />
                  <Label htmlFor={behavior} className="cursor-pointer capitalize">
                    {behavior.replaceAll("_", " ")}
                  </Label>
                </div>
                {localConfig.detection.behaviors[behavior]
                  .confidence_threshold !== undefined && (
                  <div className="flex items-center gap-2">
                    <Label className="text-sm text-muted-foreground">
                      Threshold
                    </Label>
                    <Input
                      type="number"
                      value={
                        localConfig.detection.behaviors[behavior]
                          .confidence_threshold
                      }
                      onChange={(e) =>
                        patchConfig({
                          ...localConfig,
                          detection: {
                            ...localConfig.detection,
                            behaviors: {
                              ...localConfig.detection.behaviors,
                              [behavior]: {
                                ...localConfig.detection.behaviors[behavior],
                                confidence_threshold:
                                  parseFloat(e.target.value) || 0.3,
                              },
                            },
                          },
                        })
                      }
                      className="h-8 w-20 text-sm"
                      step={0.05}
                      min={0}
                      max={1}
                    />
                  </div>
                )}
              </div>
            ))}
          </CardContent>
        </Card>

        {/* Visual alert */}
        <Card>
          <CardHeader className="pb-4">
            <CardTitle className="text-lg">Alert feedback</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-xs text-muted-foreground">
              Choose how you want to be alerted when a behavior fires.
              Enable beep, the red on-screen vignette, or both.
            </p>
            <div className="flex flex-wrap items-center gap-6">
              <div className="flex items-center gap-3">
                <Switch
                  id="alert-sound-enabled"
                  checked={localConfig.actions.sound.enabled}
                  onCheckedChange={(checked) =>
                    patchConfig({
                      ...localConfig,
                      actions: {
                        ...localConfig.actions,
                        sound: {
                          ...localConfig.actions.sound,
                          enabled: checked,
                        },
                      },
                    })
                  }
                />
                <Label htmlFor="alert-sound-enabled" className="cursor-pointer">
                  Beep
                </Label>
              </div>
              <div className="flex items-center gap-3">
                <Switch
                  id="alert-visual-enabled"
                  checked={localConfig.actions.visual?.enabled ?? true}
                  onCheckedChange={(checked) =>
                    patchConfig({
                      ...localConfig,
                      actions: {
                        ...localConfig.actions,
                        visual: {
                          ...(localConfig.actions.visual ?? { enabled: true }),
                          enabled: checked,
                        },
                      },
                    })
                  }
                />
                <Label htmlFor="alert-visual-enabled" className="cursor-pointer">
                  Red vignette
                </Label>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Sound */}
        <Card>
          <CardHeader className="pb-4">
            <CardTitle className="text-lg">Sound</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Switch
                  id="sound-repeat"
                  checked={localConfig.actions.sound.repeat}
                  onCheckedChange={(checked) =>
                    patchConfig({
                      ...localConfig,
                      actions: {
                        ...localConfig.actions,
                        sound: {
                          ...localConfig.actions.sound,
                          repeat: checked,
                        },
                      },
                    })
                  }
                />
                <Label htmlFor="sound-repeat" className="cursor-pointer">
                  Repeat
                </Label>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>Volume</Label>
                <span className="text-sm text-muted-foreground">
                  {Math.round(localConfig.actions.sound.volume * 100)}%
                </span>
              </div>
              <Slider
                value={[localConfig.actions.sound.volume]}
                onValueChange={([value]) =>
                  patchConfig({
                    ...localConfig,
                    actions: {
                      ...localConfig.actions,
                      sound: {
                        ...localConfig.actions.sound,
                        volume: value,
                      },
                    },
                  })
                }
                min={0}
                max={1}
                step={0.05}
              />
            </div>
          </CardContent>
        </Card>

        {/* Hotkeys */}
        <Card>
          <CardHeader className="pb-4">
            <CardTitle className="text-lg">Hotkeys</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-4 sm:grid-cols-3">
            <div className="space-y-2">
              <Label htmlFor="hotkey-dismiss">Dismiss (False Positive)</Label>
              <Input
                id="hotkey-dismiss"
                value={localConfig.hotkeys.dismiss_false_positive}
                onChange={(e) =>
                  updateConfig("hotkeys", {
                    dismiss_false_positive: e.target.value,
                  })
                }
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="hotkey-missed">Mark Missed</Label>
              <Input
                id="hotkey-missed"
                value={localConfig.hotkeys.mark_missed_event}
                onChange={(e) =>
                  updateConfig("hotkeys", {
                    mark_missed_event: e.target.value,
                  })
                }
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="hotkey-pause">Pause/Resume</Label>
              <Input
                id="hotkey-pause"
                value={localConfig.hotkeys.pause_resume}
                onChange={(e) =>
                  updateConfig("hotkeys", { pause_resume: e.target.value })
                }
              />
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
