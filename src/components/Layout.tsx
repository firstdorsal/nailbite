import { useEffect, useState } from "react";
import { NavLink } from "react-router-dom";
import { invoke } from "@tauri-apps/api/core";
import { cn } from "@/lib/utils";
import { ResizeEdges } from "./ResizeEdges";
import { StatusIndicator, type Status } from "./StatusIndicator";
import { TitleBar } from "./TitleBar";
import { useTheme } from "./ThemeProvider";
import { useDetection } from "@/hooks/useDetection";
import { useConfig } from "@/hooks/useConfig";
import {
  Camera,
  Pause,
  Play,
  Settings,
  History,
  LineChart,
  Sun,
  Moon,
  Monitor,
  Volume2,
  VolumeX,
} from "lucide-react";

interface LayoutProps {
  children: React.ReactNode;
}

const navItems = [
  { to: "/preview", label: "Preview", icon: Camera },
  { to: "/history", label: "History", icon: History },
  { to: "/insights", label: "Insights", icon: LineChart },
];

export default function Layout({ children }: LayoutProps) {
  const { theme, setTheme } = useTheme();
  const { alertActive, paused, present, setPaused, todayCount } = useDetection();
  const { config } = useConfig();

  const [muted, setMuted] = useState(false);

  // On mount, sync the local pause/mute state with the backend so the icons
  // reflect reality even after a route change or app restart.
  useEffect(() => {
    void invoke<{ paused: boolean; muted: boolean }>("get_runtime_state")
      .then((s) => {
        setPaused(s.paused);
        setMuted(s.muted);
      })
      .catch((e) => {
        console.warn("Failed to fetch runtime state:", e);
      });
  }, [setPaused]);

  const handleTogglePause = async () => {
    try {
      const next = await invoke<boolean>("toggle_pause");
      setPaused(next);
    } catch (e) {
      console.error("Failed to toggle pause:", e);
    }
  };

  const handleToggleMute = async () => {
    try {
      const next = await invoke<boolean>("toggle_mute");
      setMuted(next);
    } catch (e) {
      console.error("Failed to toggle mute:", e);
    }
  };

  const getStatus = (): Status => {
    if (alertActive) return "alert";
    if (paused) return "paused";
    if (!present) return "absent";
    return "normal";
  };

  const cycleTheme = () => {
    const themes: ("light" | "dark" | "system")[] = ["light", "dark", "system"];
    const currentIndex = themes.indexOf(theme);
    const nextIndex = (currentIndex + 1) % themes.length;
    setTheme(themes[nextIndex]);
  };

  const ThemeIcon = theme === "light" ? Sun : theme === "dark" ? Moon : Monitor;
  const visualEnabled = config?.actions.visual?.enabled ?? true;
  const showCount = config?.general?.show_detection_count ?? false;

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden">
      <ResizeEdges />
      <TitleBar />
      <div className="flex flex-1 overflow-hidden">
      {/* Sidebar */}
      <aside className="flex w-16 flex-col items-center border-r bg-card py-4">
        {/* Status + global controls (always visible at the top) */}
        <div className="mb-3 flex flex-col items-center gap-2">
          <div className="mb-2">
            <StatusIndicator
              status={getStatus()}
              showLabel={false}
              count={showCount ? todayCount : undefined}
            />
          </div>

          <button
            onClick={handleTogglePause}
            className={cn(
              "flex h-12 w-12 items-center justify-center rounded-lg transition-colors",
              paused
                ? "bg-yellow-500/20 text-yellow-500 hover:bg-yellow-500/30"
                : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
            )}
            title={paused ? "Resume detection" : "Pause detection"}
          >
            {paused ? (
              <Play className="h-5 w-5" />
            ) : (
              <Pause className="h-5 w-5" />
            )}
          </button>

          <button
            onClick={handleToggleMute}
            className={cn(
              "flex h-12 w-12 items-center justify-center rounded-lg transition-colors",
              muted
                ? "bg-red-500/20 text-red-500 hover:bg-red-500/30"
                : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
            )}
            title={muted ? "Unmute alert sound" : "Mute alert sound"}
          >
            {muted ? (
              <VolumeX className="h-5 w-5" />
            ) : (
              <Volume2 className="h-5 w-5" />
            )}
          </button>
        </div>

        <div className="mb-3 w-8 border-t" />

        <nav className="flex flex-1 flex-col gap-2">
          {navItems.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                cn(
                  "flex h-12 w-12 items-center justify-center rounded-lg transition-colors",
                  isActive
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                )
              }
              title={label}
            >
              <Icon className="h-5 w-5" />
            </NavLink>
          ))}
        </nav>

        <div className="mt-2 flex flex-col gap-2 border-t pt-3">
          <NavLink
            to="/settings"
            className={({ isActive }) =>
              cn(
                "flex h-12 w-12 items-center justify-center rounded-lg transition-colors",
                isActive
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
              )
            }
            title="Settings"
          >
            <Settings className="h-5 w-5" />
          </NavLink>

          <button
            onClick={cycleTheme}
            className="flex h-12 w-12 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
            title={`Theme: ${theme}`}
          >
            <ThemeIcon className="h-5 w-5" />
          </button>
        </div>
      </aside>

      {/* Main content. The alert vignette is an in-window overlay rather
          than a separate fullscreen Tauri window — Linux compositors
          don't reliably support transparent click-through windows for a
          monitor-wide effect, so we keep it scoped to this window with
          `pointer-events-none` so it never blocks UI underneath. */}
      <main className="relative flex-1 overflow-auto p-6">
        {children}
        {alertActive && visualEnabled && (
          <div
            className="pointer-events-none fixed inset-0 z-50 animate-pulse"
            style={{
              boxShadow: "inset 0 0 120px 40px rgba(239, 68, 68, 0.6)",
            }}
            aria-hidden
          />
        )}
      </main>
      </div>
    </div>
  );
}
