import { createContext, useContext, useState, type ReactNode } from "react";

interface DetectionState {
  alertActive: boolean;
  paused: boolean;
  currentBfrb: string | null;
  currentConfidence: number | null;
  setAlertActive: (active: boolean) => void;
  setPaused: (paused: boolean) => void;
  setCurrentBfrb: (bfrb: string | null) => void;
  setCurrentConfidence: (confidence: number | null) => void;
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
  const [currentBfrb, setCurrentBfrb] = useState<string | null>(null);
  const [currentConfidence, setCurrentConfidence] = useState<number | null>(
    null
  );

  return (
    <DetectionContext.Provider
      value={{
        alertActive,
        paused,
        currentBfrb,
        currentConfidence,
        setAlertActive,
        setPaused,
        setCurrentBfrb,
        setCurrentConfidence,
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
