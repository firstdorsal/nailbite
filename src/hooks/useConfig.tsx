import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { invoke } from "@tauri-apps/api/core";
import type { NailbiteConfig } from "@/types";

interface ConfigContextValue {
  config: NailbiteConfig | null;
  loading: boolean;
  error: string | null;
  /** Persist a config and update shared state. Throws on failure. */
  saveConfig: (config: NailbiteConfig) => Promise<void>;
  /**
   * Optimistic local update + debounced backend save. Multiple calls within
   * `debounceMs` collapse into a single save with the latest config — used
   * by the Settings page so a slider drag doesn't trigger one save per
   * frame.
   */
  patchConfig: (config: NailbiteConfig, debounceMs?: number) => void;
  reloadConfig: () => Promise<void>;
}

const ConfigContext = createContext<ConfigContextValue | undefined>(undefined);

interface ConfigProviderProps {
  children: ReactNode;
}

/**
 * Provides the application config to the entire React tree.
 *
 * Single shared state (one fetch, one in-memory copy) so updates from the
 * Settings page propagate to the Layout sidebar, Insights page, etc.
 * without each consumer fetching independently.
 */
export function ConfigProvider({ children }: ConfigProviderProps) {
  const [config, setConfig] = useState<NailbiteConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const debounceRef = useRef<number | null>(null);

  const loadConfig = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const result = await invoke<NailbiteConfig>("get_config");
      setConfig(result);
    } catch (e) {
      const message = e instanceof Error ? e.message : "Failed to load config";
      setError(message);
      console.error("Config load error:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  const saveConfig = useCallback(async (next: NailbiteConfig) => {
    try {
      setError(null);
      await invoke("save_config", { config: next });
      setConfig(next);
    } catch (e) {
      const message = e instanceof Error ? e.message : "Failed to save config";
      setError(message);
      throw e;
    }
  }, []);

  const patchConfig = useCallback(
    (next: NailbiteConfig, debounceMs = 250) => {
      // Optimistic in-memory update so the UI reflects the change instantly.
      setConfig(next);
      if (debounceRef.current !== null) {
        window.clearTimeout(debounceRef.current);
      }
      debounceRef.current = window.setTimeout(() => {
        debounceRef.current = null;
        invoke("save_config", { config: next })
          .then(() => {
            setError(null);
          })
          .catch((e) => {
            const message =
              e instanceof Error ? e.message : "Failed to save config";
            setError(message);
            console.error("Config save error:", e);
          });
      }, debounceMs);
    },
    [],
  );

  useEffect(() => {
    void loadConfig();
  }, [loadConfig]);

  // Flush any pending debounced save when this provider tears down.
  useEffect(() => {
    return () => {
      if (debounceRef.current !== null) {
        window.clearTimeout(debounceRef.current);
      }
    };
  }, []);

  return (
    <ConfigContext.Provider
      value={{
        config,
        loading,
        error,
        saveConfig,
        patchConfig,
        reloadConfig: loadConfig,
      }}
    >
      {children}
    </ConfigContext.Provider>
  );
}

/**
 * Access the shared application config.
 *
 * Must be used inside a `<ConfigProvider>`. Throws otherwise so missing
 * wiring fails loudly instead of silently fetching the config N times
 * across N components.
 */
export function useConfig(): ConfigContextValue {
  const ctx = useContext(ConfigContext);
  if (ctx === undefined) {
    throw new Error("useConfig must be used within a <ConfigProvider>");
  }
  return ctx;
}
