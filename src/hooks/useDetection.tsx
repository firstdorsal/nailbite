import { createContext, useContext, useState, type ReactNode } from "react";
import type { DetectionExplanation } from "@/types";

interface DetectionState {
  alertActive: boolean;
  paused: boolean;
  /**
   * Whether a user is currently in frame, as decided by the backend's
   * multi-modal presence tracker (face mesh + pose torso landmarks).
   * Drives the dark-gray "no one in frame" indicator.
   */
  present: boolean;
  currentBfrb: string | null;
  currentConfidence: number | null;
  currentExplanation: DetectionExplanation | null;
  /**
   * Base64-encoded JPEG of the trigger frame, emitted inline with the
   * `bfrb-detected` event so the modal can render an image immediately
   * on open rather than waiting for the event-history directory to
   * finalise. Replaced by the annotated frame later once `eventId`
   * resolves and event details are fetched.
   */
  triggerFrame: string | null;
  /** Detections confirmed so far today (local time). */
  todayCount: number;
  /**
   * Monotonic counter bumped whenever the alert is resolved out-of-band
   * (e.g. via the desktop notification's verdict buttons). The
   * AlertModal watches this so it can close immediately instead of
   * waiting out its 12 s linger window — the user already decided.
   */
  externalResolveSignal: number;
  setAlertActive: (active: boolean) => void;
  setPaused: (paused: boolean) => void;
  setPresent: (present: boolean) => void;
  setCurrentBfrb: (bfrb: string | null) => void;
  setCurrentConfidence: (confidence: number | null) => void;
  setCurrentExplanation: (explanation: DetectionExplanation | null) => void;
  setTriggerFrame: (frame: string | null) => void;
  setTodayCount: (count: number) => void;
  bumpExternalResolveSignal: () => void;
}

const DetectionContext = createContext<DetectionState | undefined>(undefined);

/**
 * Provider component for detection state context.
 *
 * Wrap your application in this provider to enable detection state
 * sharing across components.
 *
 * @param children - Child components that will have access to detection state
 *
 * @example
 * ```tsx
 * <DetectionProvider>
 *   <App />
 * </DetectionProvider>
 * ```
 */
export function DetectionProvider({ children }: { children: ReactNode }) {
  const [alertActive, setAlertActive] = useState(false);
  const [paused, setPaused] = useState(false);
  // Assume present at boot — the backend will quickly flip this to
  // `false` after the absent debounce if no one is actually in frame.
  // Starting `true` avoids a flash of the dark-gray indicator on every
  // launch.
  const [present, setPresent] = useState(true);
  const [currentBfrb, setCurrentBfrb] = useState<string | null>(null);
  const [currentConfidence, setCurrentConfidence] = useState<number | null>(
    null
  );
  const [currentExplanation, setCurrentExplanation] =
    useState<DetectionExplanation | null>(null);
  const [triggerFrame, setTriggerFrame] = useState<string | null>(null);
  const [todayCount, setTodayCount] = useState(0);
  const [externalResolveSignal, setExternalResolveSignal] = useState(0);
  const bumpExternalResolveSignal = () =>
    setExternalResolveSignal((n) => n + 1);

  return (
    <DetectionContext.Provider
      value={{
        alertActive,
        paused,
        present,
        currentBfrb,
        currentConfidence,
        currentExplanation,
        triggerFrame,
        todayCount,
        externalResolveSignal,
        setAlertActive,
        setPaused,
        setPresent,
        setCurrentBfrb,
        setCurrentConfidence,
        setCurrentExplanation,
        setTriggerFrame,
        setTodayCount,
        bumpExternalResolveSignal,
      }}
    >
      {children}
    </DetectionContext.Provider>
  );
}

/**
 * Hook for accessing global detection state.
 *
 * Must be used within a `DetectionProvider`. Provides access to alert state,
 * pause state, and current detection information.
 *
 * @returns Detection state and setter functions
 * @throws Error if used outside of DetectionProvider
 *
 * @example
 * ```tsx
 * const { alertActive, paused, currentBfrb } = useDetection();
 *
 * if (alertActive) {
 *   return <AlertModal bfrbType={currentBfrb} />;
 * }
 * ```
 */
export function useDetection(): DetectionState {
  const context = useContext(DetectionContext);
  if (context === undefined) {
    throw new Error(
      "useDetection must be used within a DetectionProvider. " +
      "Wrap your component tree in <DetectionProvider>."
    );
  }
  return context;
}
