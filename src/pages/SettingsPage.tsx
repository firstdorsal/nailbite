import { useState, useEffect } from "react";
import { useConfig } from "@/hooks/useConfig";
import { cn } from "@/lib/utils";
import { Save, RefreshCw, AlertTriangle, Check } from "lucide-react";
import type { NailbiteConfig } from "@/types";

export default function SettingsPage() {
  const { config, loading, error, saveConfig, reloadConfig } = useConfig();
  const [localConfig, setLocalConfig] = useState<NailbiteConfig | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saveSuccess, setSaveSuccess] = useState(false);

  // Initialize local config from loaded config
  useEffect(() => {
    if (config) {
      setLocalConfig(config);
    }
  }, [config]);

  const handleSave = async () => {
    if (!localConfig) return;

    try {
      setSaving(true);
      setSaveError(null);
      setSaveSuccess(false);
      await saveConfig(localConfig);
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 2000);
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : "Failed to save");
    } finally {
      setSaving(false);
    }
  };

  const updateConfig = <K extends keyof NailbiteConfig>(
    section: K,
    updates: Partial<NailbiteConfig[K]>
  ) => {
    if (!localConfig) return;
    setLocalConfig({
      ...localConfig,
      [section]: { ...localConfig[section], ...updates },
    });
  };

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error || !localConfig) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-4">
        <AlertTriangle className="h-12 w-12 text-destructive" />
        <p className="text-destructive">{error || "Failed to load config"}</p>
        <button
          onClick={reloadConfig}
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
        <h1 className="text-2xl font-bold">Settings</h1>
        <div className="flex gap-2">
          <button
            onClick={reloadConfig}
            className="flex items-center gap-2 rounded-lg bg-secondary px-3 py-2 text-sm"
            disabled={loading}
          >
            <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} />
            Reload
          </button>
          <button
            onClick={handleSave}
            className={cn(
              "flex items-center gap-2 rounded-lg px-3 py-2 text-sm",
              saveSuccess
                ? "bg-green-500 text-white"
                : "bg-primary text-primary-foreground"
            )}
            disabled={saving}
          >
            {saveSuccess ? (
              <Check className="h-4 w-4" />
            ) : (
              <Save className={cn("h-4 w-4", saving && "animate-spin")} />
            )}
            {saveSuccess ? "Saved" : "Save"}
          </button>
        </div>
      </div>

      {/* Save error */}
      {saveError && (
        <div className="flex items-center gap-2 rounded-lg bg-destructive/10 p-4 text-destructive">
          <AlertTriangle className="h-5 w-5" />
          <span>{saveError}</span>
        </div>
      )}

      {/* Settings sections */}
      <div className="flex-1 space-y-6 overflow-auto pb-4">
        {/* General */}
        <section className="rounded-lg border bg-card p-4">
          <h2 className="mb-4 text-lg font-semibold">General</h2>
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label className="text-sm font-medium">Log Level</label>
              <select
                value={localConfig.general.log_level}
                onChange={(e) =>
                  updateConfig("general", { log_level: e.target.value })
                }
                className="mt-1 w-full rounded-lg border bg-background px-3 py-2"
              >
                <option value="trace">Trace</option>
                <option value="debug">Debug</option>
                <option value="info">Info</option>
                <option value="warn">Warn</option>
                <option value="error">Error</option>
              </select>
            </div>
            <div>
              <label className="text-sm font-medium">Cooldown (seconds)</label>
              <input
                type="number"
                value={localConfig.general.cooldown_seconds}
                onChange={(e) =>
                  updateConfig("general", {
                    cooldown_seconds: parseInt(e.target.value) || 0,
                  })
                }
                className="mt-1 w-full rounded-lg border bg-background px-3 py-2"
                min={0}
              />
            </div>
          </div>
        </section>

        {/* Camera */}
        <section className="rounded-lg border bg-card p-4">
          <h2 className="mb-4 text-lg font-semibold">Camera</h2>
          <div className="mb-4">
            <label className="text-sm font-medium">Inference FPS</label>
            <input
              type="number"
              value={localConfig.camera.inference_fps}
              onChange={(e) =>
                updateConfig("camera", {
                  inference_fps: parseInt(e.target.value) || 8,
                })
              }
              className="mt-1 w-full rounded-lg border bg-background px-3 py-2"
              min={1}
              max={30}
            />
          </div>
          <h3 className="mb-2 text-sm font-medium">Camera Sources</h3>
          <div className="space-y-4">
            {localConfig.camera.sources.map((source, index) => (
              <div
                key={source.id}
                className="rounded-lg bg-secondary/50 p-3"
              >
                <div className="mb-2 flex items-center justify-between">
                  <span className="font-medium">{source.id}</span>
                  <span
                    className={cn(
                      "rounded px-2 py-0.5 text-xs",
                      source.role === "primary"
                        ? "bg-primary/20 text-primary"
                        : "bg-muted text-muted-foreground"
                    )}
                  >
                    {source.role}
                  </span>
                </div>
                <div className="grid gap-2 sm:grid-cols-3">
                  <div>
                    <label className="text-xs text-muted-foreground">Device</label>
                    <input
                      type="text"
                      value={source.device}
                      onChange={(e) => {
                        const newSources = [...localConfig.camera.sources];
                        newSources[index] = { ...source, device: e.target.value };
                        updateConfig("camera", { sources: newSources });
                      }}
                      className="mt-1 w-full rounded border bg-background px-2 py-1 text-sm"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-muted-foreground">Width</label>
                    <input
                      type="number"
                      value={source.resolution_width}
                      onChange={(e) => {
                        const newSources = [...localConfig.camera.sources];
                        newSources[index] = {
                          ...source,
                          resolution_width: parseInt(e.target.value) || 640,
                        };
                        updateConfig("camera", { sources: newSources });
                      }}
                      className="mt-1 w-full rounded border bg-background px-2 py-1 text-sm"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-muted-foreground">Height</label>
                    <input
                      type="number"
                      value={source.resolution_height}
                      onChange={(e) => {
                        const newSources = [...localConfig.camera.sources];
                        newSources[index] = {
                          ...source,
                          resolution_height: parseInt(e.target.value) || 480,
                        };
                        updateConfig("camera", { sources: newSources });
                      }}
                      className="mt-1 w-full rounded border bg-background px-2 py-1 text-sm"
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Detection - Behaviors */}
        <section className="rounded-lg border bg-card p-4">
          <h2 className="mb-4 text-lg font-semibold">Detection Behaviors</h2>
          <div className="space-y-4">
            {(
              ["nail_biting", "nail_picking", "hair_pulling", "skin_picking", "lip_biting"] as const
            ).map((behavior) => (
              <div
                key={behavior}
                className="flex items-center justify-between rounded-lg bg-secondary/50 p-3"
              >
                <div className="flex items-center gap-3">
                  <input
                    type="checkbox"
                    id={behavior}
                    checked={localConfig.detection.behaviors[behavior].enabled}
                    onChange={(e) =>
                      setLocalConfig({
                        ...localConfig,
                        detection: {
                          ...localConfig.detection,
                          behaviors: {
                            ...localConfig.detection.behaviors,
                            [behavior]: {
                              ...localConfig.detection.behaviors[behavior],
                              enabled: e.target.checked,
                            },
                          },
                        },
                      })
                    }
                    className="h-4 w-4 rounded border"
                  />
                  <label htmlFor={behavior} className="font-medium capitalize">
                    {behavior.replace("_", " ")}
                  </label>
                </div>
                {localConfig.detection.behaviors[behavior].confidence_threshold !==
                  undefined && (
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-muted-foreground">
                      Threshold:
                    </span>
                    <input
                      type="number"
                      value={
                        localConfig.detection.behaviors[behavior]
                          .confidence_threshold
                      }
                      onChange={(e) =>
                        setLocalConfig({
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
                      className="w-20 rounded border bg-background px-2 py-1 text-sm"
                      step={0.05}
                      min={0}
                      max={1}
                    />
                  </div>
                )}
              </div>
            ))}
          </div>
        </section>

        {/* Sound */}
        <section className="rounded-lg border bg-card p-4">
          <h2 className="mb-4 text-lg font-semibold">Sound</h2>
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="flex items-center gap-3">
              <input
                type="checkbox"
                id="sound_enabled"
                checked={localConfig.actions.sound.enabled}
                onChange={(e) =>
                  setLocalConfig({
                    ...localConfig,
                    actions: {
                      ...localConfig.actions,
                      sound: {
                        ...localConfig.actions.sound,
                        enabled: e.target.checked,
                      },
                    },
                  })
                }
                className="h-4 w-4 rounded border"
              />
              <label htmlFor="sound_enabled" className="font-medium">
                Enable Sound
              </label>
            </div>
            <div>
              <label className="text-sm font-medium">Volume</label>
              <input
                type="range"
                value={localConfig.actions.sound.volume}
                onChange={(e) =>
                  setLocalConfig({
                    ...localConfig,
                    actions: {
                      ...localConfig.actions,
                      sound: {
                        ...localConfig.actions.sound,
                        volume: parseFloat(e.target.value),
                      },
                    },
                  })
                }
                className="mt-1 w-full"
                min={0}
                max={1}
                step={0.1}
              />
              <span className="text-sm text-muted-foreground">
                {Math.round(localConfig.actions.sound.volume * 100)}%
              </span>
            </div>
            <div className="flex items-center gap-3">
              <input
                type="checkbox"
                id="sound_repeat"
                checked={localConfig.actions.sound.repeat}
                onChange={(e) =>
                  setLocalConfig({
                    ...localConfig,
                    actions: {
                      ...localConfig.actions,
                      sound: {
                        ...localConfig.actions.sound,
                        repeat: e.target.checked,
                      },
                    },
                  })
                }
                className="h-4 w-4 rounded border"
              />
              <label htmlFor="sound_repeat" className="font-medium">
                Repeat Sound
              </label>
            </div>
          </div>
        </section>

        {/* Hotkeys */}
        <section className="rounded-lg border bg-card p-4">
          <h2 className="mb-4 text-lg font-semibold">Hotkeys</h2>
          <div className="grid gap-4 sm:grid-cols-3">
            <div>
              <label className="text-sm font-medium">Dismiss (False Positive)</label>
              <input
                type="text"
                value={localConfig.hotkeys.dismiss_false_positive}
                onChange={(e) =>
                  updateConfig("hotkeys", {
                    dismiss_false_positive: e.target.value,
                  })
                }
                className="mt-1 w-full rounded-lg border bg-background px-3 py-2"
              />
            </div>
            <div>
              <label className="text-sm font-medium">Mark Missed</label>
              <input
                type="text"
                value={localConfig.hotkeys.mark_missed_event}
                onChange={(e) =>
                  updateConfig("hotkeys", { mark_missed_event: e.target.value })
                }
                className="mt-1 w-full rounded-lg border bg-background px-3 py-2"
              />
            </div>
            <div>
              <label className="text-sm font-medium">Pause/Resume</label>
              <input
                type="text"
                value={localConfig.hotkeys.pause_resume}
                onChange={(e) =>
                  updateConfig("hotkeys", { pause_resume: e.target.value })
                }
                className="mt-1 w-full rounded-lg border bg-background px-3 py-2"
              />
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
