import { useState, useEffect, useCallback, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import type { Exercise, VerificationResult, BfrbType } from "@/types";

type ExercisePhase = "idle" | "loading" | "active" | "completed" | "timeout";

interface UseExerciseOptions {
  bfrbType: BfrbType | null;
  onComplete?: () => void;
  onTimeout?: () => void;
}

interface UseExerciseResult {
  exercise: Exercise | null;
  phase: ExercisePhase;
  progress: number;
  currentRep: number;
  timeRemaining: number;
  verification: VerificationResult | null;
  error: string | null;
  startExercise: () => Promise<void>;
  verifyFrame: (frameData: number[]) => Promise<void>;
  cancelExercise: () => void;
}

/**
 * Hook for managing exercise sessions.
 *
 * Handles the full exercise lifecycle: loading an exercise from the backend,
 * tracking time/progress, verifying poses via frame data, and managing
 * completion/timeout states.
 *
 * @param options - Configuration options
 * @param options.bfrbType - The BFRB type to get an exercise for (or null to disable)
 * @param options.onComplete - Callback when exercise is completed successfully
 * @param options.onTimeout - Callback when exercise times out
 * @returns Exercise state and control functions
 *
 * @example
 * ```tsx
 * const { exercise, phase, progress, startExercise, cancelExercise } = useExercise({
 *   bfrbType: 'nail_biting',
 *   onComplete: () => console.log('Exercise completed!'),
 * });
 * ```
 */
export function useExercise(options: UseExerciseOptions): UseExerciseResult {
  const { bfrbType, onComplete, onTimeout } = options;

  const [exercise, setExercise] = useState<Exercise | null>(null);
  const [phase, setPhase] = useState<ExercisePhase>("idle");
  const [progress, setProgress] = useState(0);
  const [currentRep, setCurrentRep] = useState(0);
  const [timeRemaining, setTimeRemaining] = useState(0);
  const [verification, setVerification] = useState<VerificationResult | null>(
    null
  );
  const [error, setError] = useState<string | null>(null);

  const timerRef = useRef<number | null>(null);
  const startTimeRef = useRef<number>(0);

  const startExercise = useCallback(async () => {
    if (!bfrbType) return;

    try {
      setPhase("loading");
      setError(null);
      setProgress(0);
      setCurrentRep(0);

      const ex = await invoke<Exercise>("get_exercise", {
        bfrb_type: bfrbType,
      });

      setExercise(ex);
      setTimeRemaining(
        ex.category === "timed_hold" ? ex.hold_duration_secs : 120
      );
      startTimeRef.current = Date.now();
      setPhase("active");

      // Start countdown timer
      timerRef.current = window.setInterval(() => {
        const elapsed = (Date.now() - startTimeRef.current) / 1000;
        const remaining = Math.max(
          0,
          (ex.category === "timed_hold" ? ex.hold_duration_secs : 120) - elapsed
        );
        setTimeRemaining(Math.ceil(remaining));

        if (remaining <= 0) {
          setPhase("timeout");
          if (timerRef.current) {
            clearInterval(timerRef.current);
          }
          onTimeout?.();
        }
      }, 1000);
    } catch (e) {
      const message =
        e instanceof Error ? e.message : "Failed to start exercise";
      setError(message);
      setPhase("idle");
    }
  }, [bfrbType, onTimeout]);

  const verifyFrame = useCallback(
    async (frameData: number[]) => {
      if (!exercise || phase !== "active") return;

      try {
        const result = await invoke<VerificationResult>("verify_exercise_frame", {
          data: frameData,
          exercise_id: exercise.id,
        });

        setVerification(result);
        setProgress(result.progress);

        if (exercise.category === "repetitions" && result.pose_correct) {
          setCurrentRep((prev) => {
            const next = prev + 1;
            if (next >= exercise.target_reps) {
              setPhase("completed");
              if (timerRef.current) {
                clearInterval(timerRef.current);
              }
              onComplete?.();
            }
            return next;
          });
        } else if (exercise.category === "timed_hold" && result.progress >= 1) {
          setPhase("completed");
          if (timerRef.current) {
            clearInterval(timerRef.current);
          }
          onComplete?.();
        }
      } catch (e) {
        console.error("Verification error:", e);
      }
    },
    [exercise, phase, onComplete]
  );

  const cancelExercise = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setPhase("idle");
    setExercise(null);
    setProgress(0);
    setCurrentRep(0);
    setVerification(null);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);

  return {
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
  };
}
