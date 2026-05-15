import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";
import type {
  HandDetection,
  FaceDetection,
  PoseDetection,
  DetectionExplanation,
  HandSignal,
} from "@/types";

interface LandmarkCanvasProps {
  width: number;
  height: number;
  hands: HandDetection[];
  face: FaceDetection | null;
  pose: PoseDetection | null;
  imageRef: React.RefObject<HTMLImageElement | null>;
  showLandmarks?: boolean;
  /** Base64 encoded frame data for direct drawing (avoids async load lag) */
  frameBase64?: string;
  /**
   * Per-detector signals for the current frame. Used to overlay
   * contributing-fingertip → target lines so the user can *see* what is
   * driving the confidence value.
   */
  signals?: DetectionExplanation[];
  /** Extra classes applied to the canvas element (e.g. CSS sizing). */
  className?: string;
}

// Hand landmark connections for skeleton drawing
const HAND_CONNECTIONS = [
  // Thumb
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  // Index
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  // Middle
  [0, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  // Ring
  [0, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  // Pinky
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  // Palm
  [5, 9],
  [9, 13],
  [13, 17],
];

// Outer lip landmark indices (MediaPipe face mesh)
const OUTER_LIP_INDICES = [
  61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84,
  181, 91, 146,
];

// Inner lip landmark indices
const INNER_LIP_INDICES = [
  78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87,
  178, 88, 95,
];

// BlazePose UPPER BODY skeleton connections only
// For webcam BFRB detection, we only need face, shoulders, and arms
// Landmark indices: 0=nose, 1-6=eyes, 7-8=ears, 9-10=mouth, 11-12=shoulders,
// 13-14=elbows, 15-16=wrists, 17-22=hand landmarks
const POSE_CONNECTIONS = [
  // Face (skip detailed eye connections, just show key points)
  [0, 9], [0, 10], // Nose to mouth corners
  [9, 10], // Mouth
  // Torso (upper only)
  [11, 12], // Shoulders
  // Left arm
  [11, 13], [13, 15], // Shoulder to elbow to wrist
  // Right arm
  [12, 14], [14, 16], // Shoulder to elbow to wrist
];

// Higher visibility threshold to filter out hallucinated/low-confidence landmarks
const POSE_VISIBILITY_THRESHOLD = 0.65;

// Lip center landmarks used as the "target" for nail-biting signals.
const UPPER_LIP_CENTER = 13;
const LOWER_LIP_CENTER = 14;

/**
 * Resolves the (start, end) pixel coordinates for one signal, given the
 * detector context. Returns `null` when the data is incomplete.
 *
 * - nail_biting: contributing fingertip → mouth center
 * - nail_picking same-hand (partner on same hand_index): contrib → partner on same hand
 * - nail_picking inter-hand: contrib → partner on the other hand (matched
 *   via the paired signal that has identical normalized_distance)
 */
function resolveSignalLine(
  signal: HandSignal,
  exp: DetectionExplanation,
  hands: HandDetection[],
  face: FaceDetection | null,
  width: number,
  height: number,
): { ax: number; ay: number; bx: number; by: number } | null {
  const fromHand = hands[signal.hand_index];
  if (!fromHand || signal.contributing_fingertip == null) return null;
  const fromTip = fromHand.landmarks[signal.contributing_fingertip];
  if (!fromTip) return null;
  const ax = fromTip.x * width;
  const ay = fromTip.y * height;

  if (exp.bfrb_type === "nail_biting") {
    if (!face) return null;
    const upper = face.landmarks[UPPER_LIP_CENTER];
    const lower = face.landmarks[LOWER_LIP_CENTER];
    if (!upper || !lower) return null;
    const mx = ((upper.x + lower.x) / 2) * width;
    const my = ((upper.y + lower.y) / 2) * height;
    return { ax, ay, bx: mx, by: my };
  }

  if (signal.partner_fingertip == null) return null;

  // Same-hand picking: a paired signal with same hand_index does not exist —
  // partner is on the same hand. Inter-hand: find the paired signal that has
  // matching normalized_distance and a different hand_index.
  const partnerSignal = exp.hands.find(
    (other) =>
      other !== signal &&
      other.hand_index !== signal.hand_index &&
      Math.abs(other.normalized_distance - signal.normalized_distance) < 1e-4,
  );

  const partnerHandIndex = partnerSignal?.hand_index ?? signal.hand_index;
  const partnerHand = hands[partnerHandIndex];
  if (!partnerHand) return null;
  const partnerTip = partnerHand.landmarks[signal.partner_fingertip];
  if (!partnerTip) return null;
  return {
    ax,
    ay,
    bx: partnerTip.x * width,
    by: partnerTip.y * height,
  };
}

export function LandmarkCanvas({
  width,
  height,
  hands,
  face,
  pose,
  imageRef,
  showLandmarks = true,
  frameBase64,
  signals,
  className,
}: LandmarkCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // Cache the previously decoded image so the canvas can keep showing
  // the last good frame while a new one is decoding (avoids one-frame
  // flicker on rapid base64 swaps).
  const prevImageRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let cancelled = false;

    const drawAll = (image: HTMLImageElement | null) => {
      if (cancelled) return;
      ctx.clearRect(0, 0, width, height);
      if (image && image.complete && image.naturalWidth > 0) {
        ctx.drawImage(image, 0, 0, width, height);
      } else {
        ctx.fillStyle = "#1a1a1a";
        ctx.fillRect(0, 0, width, height);
      }
      if (!showLandmarks) return;
      drawLandmarks(ctx);
    };

    // The whole point of this effect is to ensure the image and its
    // associated landmarks land on the canvas in the SAME paint. If we
    // updated React state for "image loaded" separately from the
    // landmark draw, the canvas would briefly show last-frame image +
    // current-frame landmarks during the decode (~5-15 ms) — visible as
    // overlay drift even when everything else is smooth.
    if (!frameBase64) {
      drawAll(imageRef.current);
      return () => {
        cancelled = true;
      };
    }

    const img = new Image();
    img.src = `data:image/jpeg;base64,${frameBase64}`;
    // Prefer img.decode() (resolves after pixel data is ready) but fall
    // back to onload for older WebKit (the bundled WebKitGTK 2.44 still
    // supports decode but be defensive).
    const drawWhenReady = () => {
      if (cancelled) return;
      prevImageRef.current = img;
      drawAll(img);
    };
    if (typeof img.decode === "function") {
      img.decode().then(drawWhenReady).catch(() => {
        // Decode error: keep showing the previous good frame; landmarks
        // would drift if we drew them on no image at all.
        drawAll(prevImageRef.current);
      });
    } else {
      img.onload = drawWhenReady;
    }

    return () => {
      cancelled = true;
      img.onload = null;
    };

    function drawLandmarks(ctx: CanvasRenderingContext2D) {

    // Draw hand landmarks and skeleton
    for (const hand of hands) {
      const isLeft = hand.handedness === "left";
      const isRight = hand.handedness === "right";
      const color = isLeft ? "#00ff00" : isRight ? "#ff6600" : "#888888";

      // Draw connections
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      for (const [start, end] of HAND_CONNECTIONS) {
        const startLandmark = hand.landmarks[start];
        const endLandmark = hand.landmarks[end];
        if (startLandmark && endLandmark) {
          ctx.beginPath();
          ctx.moveTo(startLandmark.x * width, startLandmark.y * height);
          ctx.lineTo(endLandmark.x * width, endLandmark.y * height);
          ctx.stroke();
        }
      }

      // Draw landmarks
      ctx.fillStyle = color;
      for (const landmark of hand.landmarks) {
        ctx.beginPath();
        ctx.arc(landmark.x * width, landmark.y * height, 4, 0, 2 * Math.PI);
        ctx.fill();
      }

      // Draw hand label near wrist (landmark 0)
      const wrist = hand.landmarks[0];
      if (wrist) {
        const label = isLeft ? "L" : isRight ? "R" : "?";
        const labelX = wrist.x * width;
        const labelY = wrist.y * height + 25; // Below wrist

        // Draw label background
        ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
        ctx.beginPath();
        ctx.roundRect(labelX - 12, labelY - 14, 24, 20, 4);
        ctx.fill();

        // Draw label text
        ctx.fillStyle = color;
        ctx.font = "bold 14px monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(label, labelX, labelY - 4);

        // Draw confidence below label
        const confText = `${Math.round(hand.confidence * 100)}%`;
        ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
        ctx.font = "10px monospace";
        ctx.fillText(confText, labelX, labelY + 10);
      }
    }

    // Draw face lip landmarks
    if (face && face.landmarks.length > 0) {
      // Draw outer lips
      ctx.strokeStyle = "#ff00ff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < OUTER_LIP_INDICES.length; i++) {
        const idx = OUTER_LIP_INDICES[i];
        const landmark = face.landmarks[idx];
        if (landmark) {
          if (i === 0) {
            ctx.moveTo(landmark.x * width, landmark.y * height);
          } else {
            ctx.lineTo(landmark.x * width, landmark.y * height);
          }
        }
      }
      ctx.closePath();
      ctx.stroke();

      // Draw inner lips
      ctx.strokeStyle = "#ff69b4";
      ctx.beginPath();
      for (let i = 0; i < INNER_LIP_INDICES.length; i++) {
        const idx = INNER_LIP_INDICES[i];
        const landmark = face.landmarks[idx];
        if (landmark) {
          if (i === 0) {
            ctx.moveTo(landmark.x * width, landmark.y * height);
          } else {
            ctx.lineTo(landmark.x * width, landmark.y * height);
          }
        }
      }
      ctx.closePath();
      ctx.stroke();

      // Draw face label near nose tip (landmark 1) - approximate center of face
      const noseTip = face.landmarks[1];
      if (noseTip) {
        const labelX = noseTip.x * width;
        const labelY = noseTip.y * height - 60; // Above nose

        // Draw label background
        ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
        ctx.beginPath();
        ctx.roundRect(labelX - 25, labelY - 10, 50, 20, 4);
        ctx.fill();

        // Draw face label
        ctx.fillStyle = "#ff00ff";
        ctx.font = "bold 12px monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("FACE", labelX, labelY);
      }
    }

    // Draw pose skeleton (upper body only for webcam BFRB detection)
    if (pose && pose.landmarks.length > 0) {
      const poseColor = "#00ffff"; // Cyan for pose

      // Only render upper body landmarks (indices 0-22)
      // Skip legs (23-32) as they're usually not visible in webcam view
      const UPPER_BODY_MAX_INDEX = 22;

      // Draw connections first (behind landmarks)
      ctx.strokeStyle = poseColor;
      ctx.lineWidth = 2;
      for (const [startIdx, endIdx] of POSE_CONNECTIONS) {
        // Skip connections involving lower body landmarks
        if (startIdx > UPPER_BODY_MAX_INDEX || endIdx > UPPER_BODY_MAX_INDEX) continue;

        const startLm = pose.landmarks[startIdx];
        const endLm = pose.landmarks[endIdx];
        if (
          startLm &&
          endLm &&
          startLm.visibility > POSE_VISIBILITY_THRESHOLD &&
          endLm.visibility > POSE_VISIBILITY_THRESHOLD
        ) {
          ctx.beginPath();
          ctx.moveTo(startLm.x * width, startLm.y * height);
          ctx.lineTo(endLm.x * width, endLm.y * height);
          ctx.stroke();
        }
      }

      // Draw landmarks (upper body only)
      for (let i = 0; i <= UPPER_BODY_MAX_INDEX && i < pose.landmarks.length; i++) {
        const lm = pose.landmarks[i];
        if (lm.visibility > POSE_VISIBILITY_THRESHOLD) {
          // Scale point size by visibility
          const radius = 3 + lm.visibility * 2;
          ctx.fillStyle = poseColor;
          ctx.beginPath();
          ctx.arc(lm.x * width, lm.y * height, radius, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
    }

    // Draw detection signal overlays last so they sit on top.
    // For each contributing hand signal, draw a line from the contributing
    // fingertip to its target (mouth for biting, partner fingertip for picking).
    // Color/width reflects whether the signal would fire (confidence > 0).
    if (signals && signals.length > 0) {
      // Dedup inter-hand pairs — they emit two mirrored signals; the line
      // between A→B is the same as B→A.
      const drawn = new Set<string>();
      for (const exp of signals) {
        for (const sig of exp.hands) {
          if (sig.normalized_distance >= sig.distance_threshold * 1.5) {
            // Way out of range — skip clutter.
            continue;
          }
          const line = resolveSignalLine(sig, exp, hands, face, width, height);
          if (!line) continue;

          // Canonical key so we don't draw both halves of an inter-hand pair.
          const key = [
            exp.bfrb_type,
            Math.min(line.ax, line.bx).toFixed(1),
            Math.min(line.ay, line.by).toFixed(1),
            Math.max(line.ax, line.bx).toFixed(1),
            Math.max(line.ay, line.by).toFixed(1),
          ].join(":");
          if (drawn.has(key)) continue;
          drawn.add(key);

          const fired = sig.confidence > 0;
          const ratio = sig.normalized_distance / sig.distance_threshold;
          // Brighter when closer; subdued when above threshold.
          const baseColor = fired ? "#ef4444" : "#9ca3af"; // red-500 / gray-400
          const lineWidth = fired ? 3 : 1.5;

          // Halo for the contributing fingertip.
          ctx.fillStyle = fired ? "rgba(239, 68, 68, 0.35)" : "rgba(156, 163, 175, 0.25)";
          ctx.beginPath();
          ctx.arc(line.ax, line.ay, fired ? 12 : 7, 0, 2 * Math.PI);
          ctx.fill();

          // Line from contributing tip to target.
          ctx.strokeStyle = baseColor;
          ctx.lineWidth = lineWidth;
          ctx.setLineDash(fired ? [] : [4, 4]);
          ctx.beginPath();
          ctx.moveTo(line.ax, line.ay);
          ctx.lineTo(line.bx, line.by);
          ctx.stroke();
          ctx.setLineDash([]);

          // Distance label at midpoint.
          const midX = (line.ax + line.bx) / 2;
          const midY = (line.ay + line.by) / 2;
          const labelText = `${ratio.toFixed(2)}×`;
          ctx.font = "bold 11px monospace";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          // Background pill for the label
          const metrics = ctx.measureText(labelText);
          const padX = 6;
          const padY = 3;
          const labelW = metrics.width + padX * 2;
          const labelH = 16;
          ctx.fillStyle = "rgba(0, 0, 0, 0.75)";
          ctx.beginPath();
          ctx.roundRect(midX - labelW / 2, midY - labelH / 2, labelW, labelH, 4);
          ctx.fill();
          ctx.fillStyle = baseColor;
          ctx.fillText(labelText, midX, midY + padY / 2);
        }
      }
    }

    }
  }, [width, height, hands, face, pose, imageRef, showLandmarks, frameBase64, signals]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className={cn("block rounded-lg", className)}
    />
  );
}
