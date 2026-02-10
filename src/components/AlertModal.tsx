import { useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
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
import { useDetection } from "@/hooks/useDetection";
import { AlertTriangle, Dumbbell, X } from "lucide-react";

/** Formats BFRB type for display. */
function formatBfrbType(bfrbType: string | null): string {
  if (!bfrbType) return "Behavior";
  return bfrbType
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

/**
 * Alert modal shown when BFRB behavior is detected.
 * Provides options to start an exercise or dismiss (mark as false positive).
 */
export default function AlertModal() {
  const navigate = useNavigate();
  const { alertActive, currentBfrb, currentConfidence, setAlertActive } =
    useDetection();

  const handleStartExercise = useCallback(() => {
    navigate("/exercise");
    // Alert will be cleared when exercise is completed
  }, [navigate]);

  const handleDismiss = useCallback(async () => {
    try {
      await invoke("dismiss_alert");
    } catch (e) {
      console.error("Failed to dismiss alert:", e);
    }
    setAlertActive(false);
  }, [setAlertActive]);

  // Keyboard shortcut: Enter to start exercise, Escape to dismiss
  useEffect(() => {
    if (!alertActive) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Enter") {
        e.preventDefault();
        handleStartExercise();
      } else if (e.key === "Escape") {
        e.preventDefault();
        handleDismiss();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [alertActive, handleStartExercise, handleDismiss]);

  return (
    <Dialog open={alertActive} onOpenChange={(open) => !open && handleDismiss()}>
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

        <div className="py-4 text-center">
          <p className="mb-2">
            We noticed you might be engaging in a repetitive behavior.
          </p>
          <p className="text-muted-foreground">
            Would you like to do a quick decoupling exercise?
          </p>
        </div>

        <DialogFooter className="flex-col gap-2 sm:flex-row">
          <Button
            variant="outline"
            onClick={handleDismiss}
            className="flex items-center gap-2"
          >
            <X className="h-4 w-4" />
            Dismiss (False Positive)
          </Button>
          <Button
            onClick={handleStartExercise}
            className="flex items-center gap-2"
            autoFocus
          >
            <Dumbbell className="h-4 w-4" />
            Start Exercise
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
