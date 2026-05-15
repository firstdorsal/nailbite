import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor, act } from "@testing-library/react";
import type { ReactNode } from "react";
import { ConfigProvider, useConfig } from "./useConfig";
import { invoke } from "@tauri-apps/api/core";
import type { NailbiteConfig } from "@/types";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
}));

const wrapper = ({ children }: { children: ReactNode }) => (
  <ConfigProvider>{children}</ConfigProvider>
);

const mockConfig: NailbiteConfig = {
  general: {
    log_level: "info",
    show_preview: false,
    cooldown_seconds: 30,
    stats_file: "~/.local/share/nailbite/stats.jsonl",
    show_detection_count: false,
  },
  models: {
    palm_detection: "./models/palm_detection.onnx",
    hand_landmark: "./models/hand_landmark.onnx",
    hand_landmark_full: "./models/hand_landmark_full.onnx",
    hand_landmark_quality: "auto",
    face_detection: "./models/face_detection.onnx",
    face_mesh: "./models/face_mesh.onnx",
    pose_landmark: "./models/pose_landmark.onnx",
  },
  camera: {
    inference_fps: 8,
    preview_fps: 24,
    sources: [
      {
        id: "main",
        device: "/dev/video0",
        role: "primary",
        resolution_width: 640,
        resolution_height: 480,
      },
    ],
  },
  ort: {
    intra_op_num_threads: 2,
    inter_op_num_threads: 1,
    gpu: {
      preference: "auto",
      backend: "auto",
      device_id: 0,
      fp16_enable: true,
      memory_limit_mb: null,
    },
  },
  detection: {
    behaviors: {
      nail_biting: {
        enabled: true,
        proximity_threshold: 0.35,
        min_sustained_ms: 1500,
        confidence_threshold: 0.3,
      },
      nail_picking: {
        enabled: true,
        proximity_threshold: 0.25,
        min_sustained_ms: 1500,
        confidence_threshold: 0.3,
      },
      hair_pulling: { enabled: false },
      skin_picking: { enabled: false },
      lip_biting: { enabled: false },
    },
    temporal: {
      window_ms: 1500,
      positive_ratio: 0.4,
    },
    false_positive: {
      typing_suppression: true,
      chin_rest_suppression: true,
      eating_suppression: true,
    },
  },
  fusion: {
    strategy: "any",
    merge_tolerance_ms: 100,
  },
  actions: {
    sound: {
      enabled: true,
      file: "builtin",
      volume: 0.8,
      repeat: true,
    },
    webhook: {
      enabled: false,
      url: "",
      timeout_ms: 5000,
    },
    visual: {
      enabled: true,
    },
  },
  hotkeys: {
    dismiss_false_positive: "F9",
    mark_missed_event: "F10",
    pause_resume: "F11",
  },
};

describe("useConfig", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("loads config on mount", async () => {
    vi.mocked(invoke).mockResolvedValueOnce(mockConfig);

    const { result } = renderHook(() => useConfig(), { wrapper });

    expect(result.current.loading).toBe(true);
    expect(result.current.config).toBe(null);

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.config).toEqual(mockConfig);
    expect(result.current.error).toBe(null);
    expect(invoke).toHaveBeenCalledWith("get_config");
  });

  it("handles load error", async () => {
    vi.mocked(invoke).mockRejectedValueOnce(new Error("Config not found"));

    const { result } = renderHook(() => useConfig(), { wrapper });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.config).toBe(null);
    expect(result.current.error).toBe("Config not found");
  });

  it("saves config successfully", async () => {
    vi.mocked(invoke)
      .mockResolvedValueOnce(mockConfig) // Initial load
      .mockResolvedValueOnce(undefined); // Save

    const { result } = renderHook(() => useConfig(), { wrapper });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    const updatedConfig = { ...mockConfig, general: { ...mockConfig.general, log_level: "debug" } };

    await act(async () => {
      await result.current.saveConfig(updatedConfig);
    });

    expect(invoke).toHaveBeenCalledWith("save_config", { config: updatedConfig });
    expect(result.current.config?.general.log_level).toBe("debug");
    expect(result.current.error).toBe(null);
  });

  it("handles save error", async () => {
    vi.mocked(invoke)
      .mockResolvedValueOnce(mockConfig) // Initial load
      .mockRejectedValueOnce(new Error("Permission denied")); // Save

    const { result } = renderHook(() => useConfig(), { wrapper });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    await act(async () => {
      await expect(result.current.saveConfig(mockConfig)).rejects.toThrow("Permission denied");
    });

    expect(result.current.error).toBe("Permission denied");
  });

  it("reloads config", async () => {
    const updatedConfig = { ...mockConfig, general: { ...mockConfig.general, log_level: "warn" } };
    vi.mocked(invoke)
      .mockResolvedValueOnce(mockConfig) // Initial load
      .mockResolvedValueOnce(updatedConfig); // Reload

    const { result } = renderHook(() => useConfig(), { wrapper });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    await act(async () => {
      await result.current.reloadConfig();
    });

    expect(result.current.config?.general.log_level).toBe("warn");
  });
});
