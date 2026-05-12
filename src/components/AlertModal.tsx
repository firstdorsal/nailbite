import { useEffect, useCallback, useState, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { SignalsPanel } from "@/components/SignalsPanel";
import { useDetection } from "@/hooks/useDetection";
import type { Verdict } from "@/types";
import { AlertTriangle, Check, HelpCircle, X } from "lucide-react";
import { cn } from "@/lib/utils";

/** Formats BFRB type for display. */
function formatBfrbType(bfrbType: string | null): string {
  if (!bfrbType) return "Behavior";
  return bfrbType
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

interface VerdictButtonProps {
  label: string;
  /** Single-key keyboard shortcut shown on the button. */
  shortcut?: string;
  icon: React.ReactNode;
  active: boolean;
  loading: boolean;
  onClick: () => void;
}

function VerdictButton({
  label,
  shortcut,
  icon,
  active,
  loading,
  onClick,
}: VerdictButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={loading}
      className={cn(
        "relative flex flex-col items-center justify-center gap-1 rounded border px-2 py-2 text-xs transition-colors",
        active
          ? "border-primary bg-primary/10 text-primary"
          : "border-border bg-background hover:bg-accent",
        loading && "opacity-60",
      )}
    >
      {icon}
      <span>{label}</span>
      {shortcut && (
        <kbd className="absolute right-1 top-1 rounded bg-muted px-1 text-[9px] text-muted-foreground">
          {shortcut}
        </kbd>
      )}
    </button>
  );
}

/**
 * Alert modal shown when BFRB behavior is detected.
 * Provides options to start an exercise or dismiss (mark as false positive).
 */
export default function AlertModal() {
  const {
    alertActive,
    currentBfrb,
    currentConfidence,
    currentExplanation,
    externalResolveSignal,
    setAlertActive,
  } = useDetection();

  // Track whether the user already labeled this alert and what the saved
  // event id is. The id may not exist immediately (post-capture is still
  // collecting frames); we retry resolution a few times after the alert opens.
  const [eventId, setEventId] = useState<string | null>(null);
  const [verdict, setVerdict] = useState<Verdict | null>(null);
  const [savingVerdict, setSavingVerdict] = useState<Verdict | null>(null);
  // Base64 trigger frame, lazily fetched once we resolve `eventId`. We
  // hang on to the previous image across alert end events so the modal
  // keeps showing the captured detection while the linger timer counts
  // down — clearing it as soon as the live alert ended would surprise
  // the user mid-labeling.
  const [triggerImage, setTriggerImage] = useState<string | null>(null);
  const lastBfrbRef = useRef<string | null>(null);

  // Keep the modal open after the backend says the alert ended so the user
  // has time to label / dismiss. The modal is shown when EITHER the live
  // alert is active OR a recent alert is still within the linger window.
  const ALERT_LINGER_MS = 12_000;
  const [lingerUntil, setLingerUntil] = useState<number | null>(null);
  // Tracks whether we've ever observed a live alert this session. Used to
  // suppress the post-alert linger window at app startup — without it, the
  // initial `alertActive=false` would trigger a linger and the modal would
  // pop up for 12 s before any detection ever fires.
  const hasSeenAlertRef = useRef(false);
  const modalOpen = alertActive || (lingerUntil !== null && Date.now() < lingerUntil);

  // Reset verdict state when a new alert fires.
  useEffect(() => {
    if (alertActive && currentBfrb && lastBfrbRef.current !== currentBfrb) {
      lastBfrbRef.current = currentBfrb;
      setEventId(null);
      setVerdict(null);
      setSavingVerdict(null);
      setTriggerImage(null);
    }
    if (!alertActive) {
      lastBfrbRef.current = null;
    }
  }, [alertActive, currentBfrb]);

  // Once we have an eventId, fetch the annotated trigger frame so the
  // modal can show what was actually captured. We pull annotated when
  // available (landmarks drawn over the frame) so the user can see why
  // the detector fired, falling back to the raw frame.
  useEffect(() => {
    if (!eventId) return;
    let cancelled = false;
    (async () => {
      try {
        type EventDetail = {
          trigger_frame: string | null;
          trigger_frame_annotated: string | null;
        };
        const detail = await invoke<EventDetail>("get_event_details", {
          eventId,
        });
        const filename = detail.trigger_frame_annotated ?? detail.trigger_frame;
        if (!filename) return;
        const base64 = await invoke<string>("get_event_frame", {
          eventId,
          filename,
        });
        if (!cancelled) setTriggerImage(base64);
      } catch (e) {
        console.warn("Failed to load trigger frame:", e);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [eventId]);

  // When the live alert ends, start the linger timer. Clear it the moment
  // a fresh alert begins so the timer can't snap the modal shut early.
  // The very first run with `alertActive=false` at mount must NOT start a
  // linger — there's no alert to linger on.
  useEffect(() => {
    if (alertActive) {
      hasSeenAlertRef.current = true;
      setLingerUntil(null);
      return;
    }
    if (!hasSeenAlertRef.current) {
      return;
    }
    setLingerUntil(Date.now() + ALERT_LINGER_MS);
    const t = window.setTimeout(() => {
      setLingerUntil(null);
    }, ALERT_LINGER_MS);
    return () => window.clearTimeout(t);
  }, [alertActive]);

  // External resolution (e.g. user clicked the desktop notification's
  // verdict buttons) closes the modal immediately — the user already
  // labeled the event, so leaving the modal up is just stale UI.
  useEffect(() => {
    if (externalResolveSignal === 0) return;
    setLingerUntil(null);
  }, [externalResolveSignal]);

  // Try to resolve the event_id once the alert is open. Retries every 600ms
  // while the alert is active and the id has not been resolved, since the
  // event directory only appears after `frames_after` post-trigger frames.
  useEffect(() => {
    if (!alertActive || !currentBfrb || eventId) return;
    let cancelled = false;

    const tryResolve = async () => {
      try {
        const id = await invoke<string | null>("find_recent_event_for_alert", {
          bfrbType: currentBfrb,
        });
        if (!cancelled && id) {
          setEventId(id);
        }
      } catch (e) {
        console.warn("Failed to resolve alert event id:", e);
      }
    };

    void tryResolve();
    const interval = setInterval(tryResolve, 600);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [alertActive, currentBfrb, eventId]);

  const handleVerdict = useCallback(
    async (next: Verdict) => {
      setSavingVerdict(next);
      try {
        // If we don't yet have an event id, poll once more synchronously.
        let id = eventId;
        if (!id && currentBfrb) {
          try {
            id = await invoke<string | null>("find_recent_event_for_alert", {
              bfrbType: currentBfrb,
            });
            if (id) setEventId(id);
          } catch {
            // ignore — handled below
          }
        }
        if (id) {
          await invoke("set_event_verdict", {
            eventId: id,
            verdict: next,
            reason: null,
          });
        } else {
          console.warn(
            "No event id resolved for alert; verdict not persisted but closing modal",
          );
        }
        setVerdict(next);
      } catch (e) {
        console.error("Failed to save verdict:", e);
      } finally {
        setSavingVerdict(null);
      }
      // Picking any verdict means the user has decided — close the modal
      // immediately instead of forcing them to click Dismiss too. This
      // also stops the alert sound / vignette via dismiss_alert.
      try {
        await invoke("dismiss_alert");
      } catch (e) {
        console.error("Failed to dismiss alert after verdict:", e);
      }
      setAlertActive(false);
      setLingerUntil(null);
    },
    [eventId, currentBfrb, setAlertActive],
  );

  const handleDismiss = useCallback(async () => {
    try {
      await invoke("dismiss_alert");
    } catch (e) {
      console.error("Failed to dismiss alert:", e);
    }
    setAlertActive(false);
    setLingerUntil(null);
  }, [setAlertActive]);

  // Keyboard shortcuts:
  //   Enter / Escape = dismiss
  //   1 = true positive, 2 = false positive, 3 = unsure
  useEffect(() => {
    if (!modalOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't intercept while the user is typing in an input.
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.isContentEditable)
      ) {
        return;
      }
      if (e.key === "Enter" || e.key === "Escape") {
        e.preventDefault();
        handleDismiss();
      } else if (e.key === "1") {
        e.preventDefault();
        void handleVerdict("true_positive");
      } else if (e.key === "2") {
        e.preventDefault();
        void handleVerdict("false_positive");
      } else if (e.key === "3") {
        e.preventDefault();
        void handleVerdict("unsure");
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [modalOpen, handleDismiss, handleVerdict]);

  return (
    <Dialog open={modalOpen} onOpenChange={(open) => !open && handleDismiss()}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-destructive">
            <AlertTriangle className="h-5 w-5" />
            {formatBfrbType(currentBfrb)} Detected
          </DialogTitle>
          <DialogDescription>
            {currentConfidence !== null && (
              <span className="text-sm text-muted-foreground">
                Confidence: {Math.round(currentConfidence * 100)}%
              </span>
            )}
          </DialogDescription>
        </DialogHeader>

        {triggerImage && (
          <div className="overflow-hidden rounded-lg border bg-black">
            <img
              src={`data:image/jpeg;base64,${triggerImage}`}
              alt="Detection frame"
              className="block w-full"
            />
          </div>
        )}

        {currentExplanation && (
          <div className="rounded-lg border bg-muted/40 p-3">
            <SignalsPanel
              explanations={[currentExplanation]}
              inline
            />
          </div>
        )}

        {/* Verdict capture: classify this alert while it's fresh. */}
        <div className="space-y-2 rounded-lg border bg-card p-3">
          <div className="flex items-baseline justify-between gap-2">
            <span className="text-sm font-medium">Was this right?</span>
            {!eventId && (
              <span className="text-[10px] text-muted-foreground">
                saving recording…
              </span>
            )}
          </div>
          <div className="grid grid-cols-3 gap-2">
            <VerdictButton
              label="Right call"
              shortcut="1"
              icon={<Check className="h-4 w-4" />}
              active={verdict === "true_positive"}
              loading={savingVerdict === "true_positive"}
              onClick={() => handleVerdict("true_positive")}
            />
            <VerdictButton
              label="False alarm"
              shortcut="2"
              icon={<X className="h-4 w-4" />}
              active={verdict === "false_positive"}
              loading={savingVerdict === "false_positive"}
              onClick={() => handleVerdict("false_positive")}
            />
            <VerdictButton
              label="Not sure"
              shortcut="3"
              icon={<HelpCircle className="h-4 w-4" />}
              active={verdict === "unsure"}
              loading={savingVerdict === "unsure"}
              onClick={() => handleVerdict("unsure")}
            />
          </div>
        </div>

        <DialogFooter>
          <Button
            onClick={handleDismiss}
            className="flex items-center gap-2"
            autoFocus
          >
            <X className="h-4 w-4" />
            Dismiss
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
