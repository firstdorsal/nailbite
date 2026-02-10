import { useEffect, useRef, useState } from "react";
import type { HandDetection, FaceDetection, PoseDetection } from "@/types";

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

export function LandmarkCanvas({
  width,
  height,
  hands,
  face,
  pose,
  imageRef,
  showLandmarks = true,
  frameBase64,
}: LandmarkCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [loadedImage, setLoadedImage] = useState<HTMLImageElement | null>(null);

  // Load image from base64 when it changes
  useEffect(() => {
    if (!frameBase64) {
      setLoadedImage(null);
      return;
    }

    const img = new Image();
    img.onload = () => {
      setLoadedImage(img);
    };
    img.src = `data:image/jpeg;base64,${frameBase64}`;

    return () => {
      img.onload = null;
    };
  }, [frameBase64]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw image frame - prefer loaded image from base64, fallback to imageRef
    const image = loadedImage || imageRef.current;
    if (image && image.complete && image.naturalWidth > 0) {
      ctx.drawImage(image, 0, 0, width, height);
    } else {
      // Draw placeholder background when no image
      ctx.fillStyle = "#1a1a1a";
      ctx.fillRect(0, 0, width, height);
    }

    if (!showLandmarks) return;

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

  }, [width, height, hands, face, pose, imageRef, showLandmarks, loadedImage]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="rounded-lg"
    />
  );
}
