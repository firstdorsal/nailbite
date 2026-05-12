import { getCurrentWindow } from "@tauri-apps/api/window";

/**
 * Invisible resize handles around the window perimeter. Required when the
 * native window decorations are disabled (`decorations: false`) — without
 * them the user has no way to resize the window from its edges.
 *
 * Each region calls Tauri's `startResizeDragging(direction)` on
 * pointerdown, which hands the gesture off to the window manager.
 */
export function ResizeEdges() {
  const handle = (
    e: React.PointerEvent,
    direction:
      | "North"
      | "South"
      | "East"
      | "West"
      | "NorthEast"
      | "NorthWest"
      | "SouthEast"
      | "SouthWest",
  ) => {
    if (e.button !== 0) return; // primary button only
    e.preventDefault();
    void getCurrentWindow()
      .startResizeDragging(direction)
      .catch((err) => {
        console.warn("startResizeDragging failed:", err);
      });
  };

  // 4px hit areas on edges, 10px on corners. z-index keeps them above app
  // chrome but below modals (which use z-50+).
  const edge = "fixed z-40";
  return (
    <>
      {/* Edges */}
      <div
        className={`${edge} left-2 right-2 top-0 h-1 cursor-n-resize`}
        onPointerDown={(e) => handle(e, "North")}
      />
      <div
        className={`${edge} bottom-0 left-2 right-2 h-1 cursor-s-resize`}
        onPointerDown={(e) => handle(e, "South")}
      />
      <div
        className={`${edge} bottom-2 left-0 top-2 w-1 cursor-w-resize`}
        onPointerDown={(e) => handle(e, "West")}
      />
      <div
        className={`${edge} bottom-2 right-0 top-2 w-1 cursor-e-resize`}
        onPointerDown={(e) => handle(e, "East")}
      />
      {/* Corners */}
      <div
        className={`${edge} left-0 top-0 h-2 w-2 cursor-nw-resize`}
        onPointerDown={(e) => handle(e, "NorthWest")}
      />
      <div
        className={`${edge} right-0 top-0 h-2 w-2 cursor-ne-resize`}
        onPointerDown={(e) => handle(e, "NorthEast")}
      />
      <div
        className={`${edge} bottom-0 left-0 h-2 w-2 cursor-sw-resize`}
        onPointerDown={(e) => handle(e, "SouthWest")}
      />
      <div
        className={`${edge} bottom-0 right-0 h-2 w-2 cursor-se-resize`}
        onPointerDown={(e) => handle(e, "SouthEast")}
      />
    </>
  );
}
