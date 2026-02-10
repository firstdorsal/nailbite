import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useExercise } from "./useExercise";
import { invoke } from "@tauri-apps/api/core";
import type { Exercise, VerificationResult } from "@/types";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
}));

const mockTimedExercise: Exercise = {
  id: "fist_clench",
  name: "Fist Clench",
  instructions: "Make tight fists with both hands and hold for 60 seconds.",
  category: "timed_hold",
  hold_duration_secs: 60,
  target_reps: 0,
};

const mockRepExercise: Exercise = {
  id: "ear_touch",
  name: "Ear Touch",
  instructions: "Touch your ear 10 times.",
  category: "repetitions",
  hold_duration_secs: 0,
  target_reps: 10,
};

const mockVerificationCorrect: VerificationResult = {
  pose_correct: true,
  progress: 0.5,
  feedback: "Good form!",
};

const mockVerificationComplete: VerificationResult = {
  pose_correct: true,
  progress: 1.0,
  feedback: "Complete!",
};

describe("useExercise", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("starts in idle phase", () => {
    const { result } = renderHook(() =>
      useExercise({ bfrbType: "nail_biting" })
    );

    expect(result.current.phase).toBe("idle");
    expect(result.current.exercise).toBe(null);
    expect(result.current.progress).toBe(0);
  });

  it("loads exercise on startExercise", async () => {
    vi.mocked(invoke).mockResolvedValueOnce(mockTimedExercise);

    const { result } = renderHook(() =>
      useExercise({ bfrbType: "nail_biting" })
    );

    await act(async () => {
      await result.current.startExercise();
    });

    expect(invoke).toHaveBeenCalledWith("get_exercise", {
      bfrb_type: "nail_biting",
    });
    expect(result.current.exercise).toEqual(mockTimedExercise);
    expect(result.current.phase).toBe("active");
    expect(result.current.timeRemaining).toBe(60);
  });

  it("handles exercise load error", async () => {
    vi.mocked(invoke).mockRejectedValueOnce(new Error("No exercises available"));

    const { result } = renderHook(() =>
      useExercise({ bfrbType: "nail_biting" })
    );

    await act(async () => {
      await result.current.startExercise();
    });

    expect(result.current.phase).toBe("idle");
    expect(result.current.error).toBe("No exercises available");
  });

  it("does nothing when bfrbType is null", async () => {
    const { result } = renderHook(() => useExercise({ bfrbType: null }));

    await act(async () => {
      await result.current.startExercise();
    });

    expect(invoke).not.toHaveBeenCalled();
    expect(result.current.phase).toBe("idle");
  });

  it("cancels exercise correctly", async () => {
    vi.mocked(invoke).mockResolvedValueOnce(mockTimedExercise);

    const { result } = renderHook(() =>
      useExercise({ bfrbType: "nail_biting" })
    );

    await act(async () => {
      await result.current.startExercise();
    });

    expect(result.current.phase).toBe("active");

    act(() => {
      result.current.cancelExercise();
    });

    expect(result.current.phase).toBe("idle");
    expect(result.current.exercise).toBe(null);
    expect(result.current.progress).toBe(0);
  });

  it("completes timed exercise on 100% progress", async () => {
    vi.mocked(invoke)
      .mockResolvedValueOnce(mockTimedExercise)
      .mockResolvedValueOnce(mockVerificationComplete);

    const onComplete = vi.fn();
    const { result } = renderHook(() =>
      useExercise({ bfrbType: "nail_biting", onComplete })
    );

    await act(async () => {
      await result.current.startExercise();
    });

    await act(async () => {
      await result.current.verifyFrame([1, 2, 3]);
    });

    expect(result.current.phase).toBe("completed");
    expect(onComplete).toHaveBeenCalled();
  });

  it("tracks rep count for repetition exercises", async () => {
    vi.mocked(invoke)
      .mockResolvedValueOnce(mockRepExercise)
      .mockResolvedValue(mockVerificationCorrect);

    const { result } = renderHook(() =>
      useExercise({ bfrbType: "nail_biting" })
    );

    await act(async () => {
      await result.current.startExercise();
    });

    expect(result.current.currentRep).toBe(0);

    await act(async () => {
      await result.current.verifyFrame([1, 2, 3]);
    });

    expect(result.current.currentRep).toBe(1);
  });

  it("cleans up timer on unmount", async () => {
    vi.mocked(invoke).mockResolvedValueOnce(mockTimedExercise);

    const { result, unmount } = renderHook(() =>
      useExercise({ bfrbType: "nail_biting" })
    );

    await act(async () => {
      await result.current.startExercise();
    });

    expect(result.current.phase).toBe("active");

    unmount();
    // Timer should be cleaned up without errors
  });
});
