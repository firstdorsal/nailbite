import { Routes, Route, Navigate } from "react-router-dom";
import { useEffect, useRef } from "react";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import Layout from "./components/Layout";
import AlertModal from "./components/AlertModal";
import PreviewPage from "./pages/PreviewPage";
import ExercisePage from "./pages/ExercisePage";
import SettingsPage from "./pages/SettingsPage";
import StatsPage from "./pages/StatsPage";
import { DetectionProvider, useDetection } from "./hooks/useDetection";

function AppRoutes() {
  const { setAlertActive, setCurrentBfrb, setCurrentConfidence } =
    useDetection();

  // Store unlisten functions in refs to ensure proper cleanup (TECH-5)
  const unlistenDetectedRef = useRef<UnlistenFn | null>(null);
  const unlistenEndedRef = useRef<UnlistenFn | null>(null);

  useEffect(() => {
    // Listen for BFRB detection events from backend
    listen<{ bfrb_type: string; confidence: number }>(
      "bfrb-detected",
      (event) => {
        setAlertActive(true);
        setCurrentBfrb(event.payload.bfrb_type);
        setCurrentConfidence(event.payload.confidence);
        // Sound is handled by Rust backend
      }
    ).then((fn) => {
      unlistenDetectedRef.current = fn;
    });

    listen<{ bfrb_type: string }>(
      "alert-ended",
      (_event) => {
        setAlertActive(false);
        setCurrentBfrb(null);
        setCurrentConfidence(null);
      }
    ).then((fn) => {
      unlistenEndedRef.current = fn;
    });

    return () => {
      unlistenDetectedRef.current?.();
      unlistenEndedRef.current?.();
    };
  }, [setAlertActive, setCurrentBfrb, setCurrentConfidence]);

  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Navigate to="/preview" replace />} />
        <Route path="/preview" element={<PreviewPage />} />
        <Route path="/exercise" element={<ExercisePage />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="/stats" element={<StatsPage />} />
      </Routes>
      <AlertModal />
    </Layout>
  );
}

export default function App() {
  return (
    <DetectionProvider>
      <AppRoutes />
    </DetectionProvider>
  );
}
