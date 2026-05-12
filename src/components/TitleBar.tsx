import { useEffect, useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { Minus, X } from "lucide-react";
import { cn } from "@/lib/utils";

/** Plain square outline used for the maximize button (Windows convention). */
function MaximizeIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 12 12"
      className={className}
      fill="none"
      stroke="currentColor"
      strokeWidth="1.25"
      aria-hidden
    >
      <rect x="1.5" y="1.5" width="9" height="9" />
    </svg>
  );
}

/** Two overlapping squares used for the restore button (Windows convention). */
function RestoreIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 12 12"
      className={className}
      fill="none"
      stroke="currentColor"
      strokeWidth="1.25"
      aria-hidden
    >
      <rect x="3" y="3" width="7" height="7" />
      <path d="M2 9V2h7" />
    </svg>
  );
}

/**
 * VS Code-style custom title bar shown when the native window decorations
 * are turned off (`tauri.conf.json: app.windows[0].decorations = false`).
 *
 * Behavior:
 *  - The bar itself is a `data-tauri-drag-region`, so click+drag moves the
 *    window. Children with `data-tauri-drag-region` participate in dragging;
 *    interactive elements (buttons) explicitly opt out so clicks land cleanly.
 *  - Double-click on the bar toggles maximize, matching most desktops.
 *  - Window control buttons sit on the right (Linux/Windows convention).
 */
export function TitleBar() {
  const win = getCurrentWindow();
  const [isMaximized, setIsMaximized] = useState(false);

  useEffect(() => {
    // Track maximized state to swap the icon between Maximize ↔ Restore.
    let cancelled = false;
    const sync = async () => {
      try {
        const m = await win.isMaximized();
        if (!cancelled) setIsMaximized(m);
      } catch {
        // ignore
      }
    };
    void sync();
    const unlistenP = win.onResized(() => {
      void sync();
    });
    return () => {
      cancelled = true;
      void unlistenP.then((u) => u());
    };
  }, [win]);

  const handleMinimize = () => {
    void win.minimize();
  };
  const handleMaximize = () => {
    void win.toggleMaximize();
  };
  const handleClose = () => {
    void win.close();
  };

  return (
    // `data-tauri-drag-region` already handles single-click drag AND
    // double-click toggle-maximize in the Tauri webview. Adding our own
    // onDoubleClick produced a second toggle on the same gesture, which
    // looked like the window briefly maximizing and snapping back.
    <div
      data-tauri-drag-region
      className="flex h-9 shrink-0 items-center justify-between border-b bg-card text-card-foreground select-none"
    >
      <div
        data-tauri-drag-region
        className="flex h-full items-center gap-2 pl-3 text-xs font-medium tracking-wide"
      >
        <span className="inline-block h-2.5 w-2.5 rounded-full bg-primary" />
        Nailbite
      </div>

      <div className="flex h-full items-stretch">
        <TitleBarButton onClick={handleMinimize} ariaLabel="Minimize">
          <Minus className="h-3.5 w-3.5" />
        </TitleBarButton>
        <TitleBarButton onClick={handleMaximize} ariaLabel={isMaximized ? "Restore" : "Maximize"}>
          {isMaximized ? (
            <RestoreIcon className="h-3 w-3" />
          ) : (
            <MaximizeIcon className="h-3 w-3" />
          )}
        </TitleBarButton>
        <TitleBarButton
          onClick={handleClose}
          ariaLabel="Close"
          className="hover:bg-destructive hover:text-destructive-foreground"
        >
          <X className="h-3.5 w-3.5" />
        </TitleBarButton>
      </div>
    </div>
  );
}

interface TitleBarButtonProps {
  onClick: () => void;
  ariaLabel: string;
  children: React.ReactNode;
  className?: string;
}

function TitleBarButton({
  onClick,
  ariaLabel,
  children,
  className,
}: TitleBarButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={ariaLabel}
      title={ariaLabel}
      className={cn(
        "flex h-full w-11 items-center justify-center text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground",
        className,
      )}
    >
      {children}
    </button>
  );
}
