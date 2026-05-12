import { Routes, Route, Navigate } from "react-router-dom";
import { useEffect, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import Layout from "./components/Layout";
import AlertModal from "./components/AlertModal";
import PreviewPage from "./pages/PreviewPage";
import SettingsPage from "./pages/SettingsPage";
import EventHistoryPage from "./pages/EventHistoryPage";
import InsightsPage from "./pages/InsightsPage";
import { ConfigProvider } from "./hooks/useConfig";
import { DetectionProvider, useDetection } from "./hooks/useDetection";

function AppRoutes() {
  const {
    setAlertActive,
    setCurrentBfrb,
    setCurrentConfidence,
    setCurrentExplanation,
    setTodayCount,
    bumpExternalResolveSignal,
  } = useDetection();

  // Store unlisten functions in refs to ensure proper cleanup (TECH-5)
  const unlistenDetectedRef = useRef<UnlistenFn | null>(null);
  const unlistenEndedRef = useRef<UnlistenFn | null>(null);
  const unlistenCountRef = useRef<UnlistenFn | null>(null);

  // Start the detection backend once on app mount and leave it running for
  // the lifetime of the app so behavior is detected from any tab — not just
  // while the Preview page is open. `start_camera` is idempotent on the
  // backend; calling it again is a no-op while detection is running.
  useEffect(() => {
    void invoke("start_camera").catch((e) => {
      console.error("Failed to start camera at app mount:", e);
    });
    // Seed today's detection count from the backend so the badge is
    // populated even before the first new detection of this session.
    void invoke<number>("get_today_detection_count")
      .then((n) => setTodayCount(n))
      .catch(() => {});
    // No cleanup: we want detection to outlive route changes. Camera is
    // stopped only when the app shuts down (Tauri tears down the process).
  }, [setTodayCount]);

  useEffect(() => {
    // Listen for BFRB detection events from backend
    listen<{
      bfrb_type: string;
      confidence: number;
      explanation?: import("./types").DetectionExplanation | null;
    }>("bfrb-detected", (event) => {
      setAlertActive(true);
      setCurrentBfrb(event.payload.bfrb_type);
      setCurrentConfidence(event.payload.confidence);
      setCurrentExplanation(event.payload.explanation ?? null);
      // Sound is handled by Rust backend
    }).then((fn) => {
      unlistenDetectedRef.current = fn;
    });

    listen<{ bfrb_type: string }>(
      "alert-ended",
      (event) => {
        setAlertActive(false);
        // Deliberately keep `currentBfrb`, `currentConfidence`, and
        // `currentExplanation` populated — the AlertModal lingers for
        // 12 s after the live alert ends, and the user expects to see
        // the captured detection image / signals during that window so
        // they can decide whether to label it. The next `bfrb-detected`
        // event will overwrite these with fresh data.
        if (event.payload.bfrb_type === "notification_action") {
          // The notification path means the user already decided — close
          // the modal immediately instead of waiting out the linger.
          bumpExternalResolveSignal();
        }
      }
    ).then((fn) => {
      unlistenEndedRef.current = fn;
    });

    listen<{ count: number }>("detection-count", (event) => {
      setTodayCount(event.payload.count);
    }).then((fn) => {
      unlistenCountRef.current = fn;
    });

    return () => {
      unlistenDetectedRef.current?.();
      unlistenEndedRef.current?.();
      unlistenCountRef.current?.();
    };
  }, [
    setAlertActive,
    setCurrentBfrb,
    setCurrentConfidence,
    setCurrentExplanation,
    setTodayCount,
    bumpExternalResolveSignal,
  ]);

  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Navigate to="/preview" replace />} />
        <Route path="/preview" element={<PreviewPage />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="/history" element={<EventHistoryPage />} />
        <Route path="/insights" element={<InsightsPage />} />
      </Routes>
      <AlertModal />
    </Layout>
  );
}

export default function App() {
  return (
    <ConfigProvider>
      <DetectionProvider>
        <AppRoutes />
      </DetectionProvider>
    </ConfigProvider>
  );
}
