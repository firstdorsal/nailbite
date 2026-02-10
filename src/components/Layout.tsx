import { NavLink } from "react-router-dom";
import { cn } from "@/lib/utils";
import { StatusIndicator, type Status } from "./StatusIndicator";
import { useTheme } from "./ThemeProvider";
import { useDetection } from "@/hooks/useDetection";
import {
  Camera,
  Dumbbell,
  Settings,
  BarChart3,
  Sun,
  Moon,
  Monitor,
} from "lucide-react";

interface LayoutProps {
  children: React.ReactNode;
}

const navItems = [
  { to: "/preview", label: "Preview", icon: Camera },
  { to: "/exercise", label: "Exercise", icon: Dumbbell },
  { to: "/settings", label: "Settings", icon: Settings },
  { to: "/stats", label: "Stats", icon: BarChart3 },
];

export default function Layout({ children }: LayoutProps) {
  const { theme, setTheme } = useTheme();
  const { alertActive, paused } = useDetection();

  const getStatus = (): Status => {
    if (alertActive) return "alert";
    if (paused) return "paused";
    return "normal";
  };

  const cycleTheme = () => {
    const themes: ("light" | "dark" | "system")[] = ["light", "dark", "system"];
    const currentIndex = themes.indexOf(theme);
    const nextIndex = (currentIndex + 1) % themes.length;
    setTheme(themes[nextIndex]);
  };

  const ThemeIcon = theme === "light" ? Sun : theme === "dark" ? Moon : Monitor;

  return (
    <div className="flex h-screen w-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="flex w-16 flex-col items-center border-r bg-card py-4">
        <div className="mb-6">
          <StatusIndicator status={getStatus()} showLabel={false} />
        </div>

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

        <button
          onClick={cycleTheme}
          className="flex h-12 w-12 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
          title={`Theme: ${theme}`}
        >
          <ThemeIcon className="h-5 w-5" />
        </button>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto p-6">{children}</main>
    </div>
  );
}
