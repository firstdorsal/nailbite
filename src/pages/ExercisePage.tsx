import { useEffect, useCallback, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { invoke } from "@tauri-apps/api/core";
import { useDetection } from "@/hooks/useDetection";
import { useExercise } from "@/hooks/useExercise";
import { useCamera } from "@/hooks/useCamera";
import { LandmarkCanvas } from "@/components/LandmarkCanvas";
import { cn } from "@/lib/utils";
import {
  CheckCircle,
  Clock,
  AlertTriangle,
  RefreshCw,
  XCircle,
} from "lucide-react";
import type { BfrbType } from "@/types";

export default function ExercisePage() {
  const navigate = useNavigate();
  const { alertActive, currentBfrb, setAlertActive } = useDetection();

  const handleComplete = useCallback(async () => {
    try {
      await invoke("dismiss_alert");
      setAlertActive(false);
      navigate("/preview");
    } catch (e) {
      console.error("Failed to complete exercise:", e);
    }
  }, [navigate, setAlertActive]);

  const handleTimeout = useCallback(() => {
    // On timeout, let user try again or dismiss
  }, []);

  const {
    exercise,
    phase,
    progress,
    currentRep,
    timeRemaining,
    verification,
    error,
    startExercise,
    verifyFrame,
    cancelExercise,
  } = useExercise({
    bfrbType: currentBfrb as BfrbType | null,
    onComplete: handleComplete,
    onTimeout: handleTimeout,
  });

  const { primaryCamera, isStreaming, getImageRef } = useCamera({
    enabled: phase === "active",
  });

  const imageRef = useRef<HTMLImageElement | null>(null);
  const primaryCameraId = primaryCamera?.cameraId ?? "main";

  // Sync with the hook's image ref
  useEffect(() => {
    const hookRef = getImageRef(primaryCameraId);
    hookRef.current = imageRef.current;
  }, [primaryCameraId, getImageRef]);

  // Start exercise when alert is active
  useEffect(() => {
    if (alertActive && currentBfrb && phase === "idle") {
      startExercise();
    }
  }, [alertActive, currentBfrb, phase, startExercise]);

  // Verify exercise poses using Rust backend's current frame
  useEffect(() => {
    if (phase !== "active" || !isStreaming || !exercise) return;

    // The Rust backend already has the current frame from camera capture.
    // We can call verify_exercise_current_frame which uses the buffered frame.
    const interval = setInterval(async () => {
      try {
        // Call the Rust command to verify the current camera frame
        await verifyFrame([]);
      } catch (e) {
        console.error("Verification error:", e);
      }
    }, 500); // Verify every 500ms

    return () => clearInterval(interval);
  }, [phase, isStreaming, exercise, verifyFrame]);

  const handleSkip = async () => {
    try {
      await invoke("dismiss_alert");
      cancelExercise();
      setAlertActive(false);
      navigate("/preview");
    } catch (e) {
      console.error("Failed to skip:", e);
    }
  };

  // No alert, redirect to preview
  if (!alertActive && phase === "idle") {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-4">
        <CheckCircle className="h-16 w-16 text-green-500" />
        <h2 className="text-2xl font-bold">No Active Alert</h2>
        <p className="text-muted-foreground">
          You&apos;ll be redirected here when a BFRB is detected.
        </p>
        <button
          onClick={() => navigate("/preview")}
          className="rounded-lg bg-primary px-4 py-2 text-primary-foreground"
        >
          Go to Preview
        </button>
      </div>
    );
  }

  const hands = primaryCamera?.hands ?? [];
  const face = primaryCamera?.face ?? null;
  const pose = primaryCamera?.pose ?? null;

  return (
    <div className="flex h-full flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">
            {exercise?.name || "Loading Exercise..."}
          </h1>
          {currentBfrb && (
            <p className="text-sm text-muted-foreground">
              Detected: {currentBfrb.replace("_", " ")}
            </p>
          )}
        </div>

        <div className="flex items-center gap-4">
          {/* Timer */}
          <div className="flex items-center gap-2 rounded-lg bg-secondary px-4 py-2">
            <Clock className="h-5 w-5" />
            <span className="font-mono text-lg">
              {Math.floor(timeRemaining / 60)}:
              {(timeRemaining % 60).toString().padStart(2, "0")}
            </span>
          </div>

          {/* Skip button */}
          <button
            onClick={handleSkip}
            className="flex items-center gap-2 rounded-lg bg-destructive/10 px-4 py-2 text-destructive hover:bg-destructive/20"
          >
            <XCircle className="h-5 w-5" />
            Skip
          </button>
        </div>
      </div>

      {/* Error display */}
      {error && (
        <div className="flex items-center gap-2 rounded-lg bg-destructive/10 p-4 text-destructive">
          <AlertTriangle className="h-5 w-5" />
          <span>{error}</span>
        </div>
      )}

      {/* Main content */}
      <div className="flex flex-1 gap-4">
        {/* Video feed */}
        <div className="relative flex-1">
          {/* Hidden image element that receives frame data */}
          <img
            ref={imageRef}
            alt="Camera feed"
            className="hidden"
            width={640}
            height={480}
          />

          {isStreaming && primaryCamera ? (
            <LandmarkCanvas
              width={640}
              height={480}
              hands={hands}
              face={face}
              pose={pose}
              imageRef={imageRef}
              showLandmarks={true}
            />
          ) : (
            <div className="flex h-[480px] w-[640px] items-center justify-center rounded-lg bg-muted">
              <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          )}

          {/* Verification overlay */}
          {verification && (
            <div
              className={cn(
                "absolute bottom-4 left-4 right-4 rounded-lg p-4",
                verification.pose_correct
                  ? "bg-green-500/90 text-white"
                  : "bg-yellow-500/90 text-white"
              )}
            >
              <p className="font-medium">{verification.feedback}</p>
            </div>
          )}
        </div>

        {/* Instructions panel */}
        <div className="flex w-80 flex-col gap-4">
          {/* Instructions */}
          <div className="rounded-lg border bg-card p-4">
            <h3 className="mb-2 font-semibold">Instructions</h3>
            <p className="whitespace-pre-line text-sm text-muted-foreground">
              {exercise?.instructions || "Loading..."}
            </p>
          </div>

          {/* Progress */}
          <div className="rounded-lg border bg-card p-4">
            <h3 className="mb-2 font-semibold">Progress</h3>

            {exercise?.category === "timed_hold" ? (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Hold progress</span>
                  <span>{Math.round(progress * 100)}%</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-secondary">
                  <div
                    className="h-full bg-primary transition-all"
                    style={{ width: `${progress * 100}%` }}
                  />
                </div>
              </div>
            ) : (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Repetitions</span>
                  <span>
                    {currentRep} / {exercise?.target_reps || 0}
                  </span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-secondary">
                  <div
                    className="h-full bg-primary transition-all"
                    style={{
                      width: `${(currentRep / (exercise?.target_reps || 1)) * 100}%`,
                    }}
                  />
                </div>
              </div>
            )}
          </div>

          {/* Phase indicator */}
          <div className="rounded-lg border bg-card p-4">
            <h3 className="mb-2 font-semibold">Status</h3>
            <div
              className={cn(
                "rounded-lg px-3 py-2 text-center text-sm font-medium",
                phase === "active" && "bg-blue-500/20 text-blue-500",
                phase === "completed" && "bg-green-500/20 text-green-500",
                phase === "timeout" && "bg-yellow-500/20 text-yellow-500",
                phase === "loading" && "bg-secondary text-secondary-foreground"
              )}
            >
              {phase === "loading" && "Loading exercise..."}
              {phase === "active" && "Exercise in progress"}
              {phase === "completed" && "Exercise completed!"}
              {phase === "timeout" && "Time's up - try again"}
            </div>

            {phase === "timeout" && (
              <button
                onClick={startExercise}
                className="mt-2 flex w-full items-center justify-center gap-2 rounded-lg bg-primary px-4 py-2 text-primary-foreground"
              >
                <RefreshCw className="h-4 w-4" />
                Try Again
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
