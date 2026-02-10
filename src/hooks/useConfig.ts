import { useState, useEffect, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import type { NailbiteConfig } from "@/types";

interface UseConfigResult {
  config: NailbiteConfig | null;
  loading: boolean;
  error: string | null;
  saveConfig: (config: NailbiteConfig) => Promise<void>;
  reloadConfig: () => Promise<void>;
}

/**
 * Hook for managing application configuration.
 *
 * Loads the configuration from the Rust backend on mount and provides
 * functions to save and reload the configuration.
 *
 * @returns Configuration state and control functions
 *
 * @example
 * ```tsx
 * const { config, loading, error, saveConfig } = useConfig();
 *
 * if (loading) return <Spinner />;
 * if (error) return <Error message={error} />;
 *
 * return <SettingsForm config={config} onSave={saveConfig} />;
 * ```
 */
export function useConfig(): UseConfigResult {
  const [config, setConfig] = useState<NailbiteConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadConfig = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const result = await invoke<NailbiteConfig>("get_config");
      setConfig(result);
    } catch (e) {
      const message =
        e instanceof Error ? e.message : "Failed to load config";
      setError(message);
      console.error("Config load error:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  const saveConfig = useCallback(async (newConfig: NailbiteConfig) => {
    try {
      setError(null);
      await invoke("save_config", { config: newConfig });
      setConfig(newConfig);
    } catch (e) {
      const message =
        e instanceof Error ? e.message : "Failed to save config";
      setError(message);
      throw e;
    }
  }, []);

  useEffect(() => {
    loadConfig();
  }, [loadConfig]);

  return {
    config,
    loading,
    error,
    saveConfig,
    reloadConfig: loadConfig,
  };
}
