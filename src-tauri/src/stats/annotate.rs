//! Frame annotation — draws hand / face / pose landmarks on top of a frame.
//!
//! Produces a copy of the input `Frame` with landmark dots, hand skeleton
//! lines, the outer-lip ring, and the upper-body pose skeleton rendered
//! directly into the pixels. Used by the event-history recorder to save
//! `frame_*_annotated.jpg` files alongside the raw captures so the UI can
//! show what the detector was looking at.
//!
//! Uses only the `image` crate; line and circle drawing are implemented
//! locally (Bresenham / brute-force) to avoid an extra dependency for what
//! is ultimately ~50 lines of pixel math.

use image::{ImageBuffer, Rgb};

use crate::detection::types::{FaceDetection, HandDetection, HandSide, Landmark, PoseDetection};
use crate::frame::Frame;

/// MediaPipe hand-landmark skeleton edges (21 landmarks per hand).
const HAND_CONNECTIONS: &[(usize, usize)] = &[
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
];

/// Outer-lip ring indices from MediaPipe face mesh (468 points).
const OUTER_LIP_INDICES: &[usize] = &[
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84,
    181, 91, 146,
];

/// BlazePose upper-body skeleton (subset of the 33-point full skeleton).
const POSE_CONNECTIONS: &[(usize, usize)] = &[
    (0, 9), (0, 10), (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
];

const POSE_VISIBILITY_THRESHOLD: f32 = 0.65;

type RgbBuf = ImageBuffer<Rgb<u8>, Vec<u8>>;

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn project(lm: &Landmark, width: u32, height: u32) -> (i32, i32) {
    let x = (lm.x * width as f32).clamp(0.0, (width - 1) as f32) as i32;
    let y = (lm.y * height as f32).clamp(0.0, (height - 1) as f32) as i32;
    (x, y)
}

fn put_pixel_safe(buf: &mut RgbBuf, x: i32, y: i32, color: Rgb<u8>) {
    if x < 0 || y < 0 {
        return;
    }
    #[allow(clippy::cast_sign_loss)]
    let (xu, yu) = (x as u32, y as u32);
    if xu >= buf.width() || yu >= buf.height() {
        return;
    }
    buf.put_pixel(xu, yu, color);
}

/// Filled circle by brute force over the bounding box. Radius is small
/// (typically ≤ 4 px) so this is cheap.
fn draw_filled_circle(buf: &mut RgbBuf, cx: i32, cy: i32, r: i32, color: Rgb<u8>) {
    let r2 = r * r;
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r2 {
                put_pixel_safe(buf, cx + dx, cy + dy, color);
            }
        }
    }
}

/// Bresenham line drawing.
fn draw_line(buf: &mut RgbBuf, x0: i32, y0: i32, x1: i32, y1: i32, color: Rgb<u8>) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut x = x0;
    let mut y = y0;
    loop {
        put_pixel_safe(buf, x, y, color);
        // Make 2-px thick lines so they read on top of JPEG noise.
        put_pixel_safe(buf, x + 1, y, color);
        put_pixel_safe(buf, x, y + 1, color);
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

fn draw_hand(buf: &mut RgbBuf, hand: &HandDetection, color: Rgb<u8>) {
    let (w, h) = buf.dimensions();

    for &(a, b) in HAND_CONNECTIONS {
        let Some(la) = hand.landmarks.get(a) else { continue };
        let Some(lb) = hand.landmarks.get(b) else { continue };
        let (ax, ay) = project(la, w, h);
        let (bx, by) = project(lb, w, h);
        draw_line(buf, ax, ay, bx, by, color);
    }

    for lm in hand.landmarks.iter() {
        let (x, y) = project(lm, w, h);
        draw_filled_circle(buf, x, y, 3, color);
    }
}

fn draw_face(buf: &mut RgbBuf, face: &FaceDetection) {
    let (w, h) = buf.dimensions();
    let color = Rgb([255, 0, 255]);

    for i in 0..OUTER_LIP_INDICES.len() {
        let Some(&a_idx) = OUTER_LIP_INDICES.get(i) else { continue };
        let Some(&b_idx) = OUTER_LIP_INDICES.get((i + 1) % OUTER_LIP_INDICES.len()) else {
            continue;
        };
        let Some(la) = face.landmarks.get(a_idx) else { continue };
        let Some(lb) = face.landmarks.get(b_idx) else { continue };
        let (ax, ay) = project(la, w, h);
        let (bx, by) = project(lb, w, h);
        draw_line(buf, ax, ay, bx, by, color);
    }
}

fn draw_pose(buf: &mut RgbBuf, pose: &PoseDetection) {
    let (w, h) = buf.dimensions();
    let color = Rgb([0, 255, 255]);

    for &(a, b) in POSE_CONNECTIONS {
        let Some(la) = pose.landmarks.get(a) else { continue };
        let Some(lb) = pose.landmarks.get(b) else { continue };
        if la.visibility < POSE_VISIBILITY_THRESHOLD || lb.visibility < POSE_VISIBILITY_THRESHOLD {
            continue;
        }
        let (ax, ay) = project(&la.landmark, w, h);
        let (bx, by) = project(&lb.landmark, w, h);
        draw_line(buf, ax, ay, bx, by, color);
    }

    for pl in pose.landmarks.iter().take(17) {
        if pl.visibility < POSE_VISIBILITY_THRESHOLD {
            continue;
        }
        let (x, y) = project(&pl.landmark, w, h);
        draw_filled_circle(buf, x, y, 3, color);
    }
}

/// Annotate a frame with hand / face / pose landmarks, returning a new
/// `Frame` whose pixels include the drawn overlays.
#[must_use]
pub fn annotate_frame(
    frame: &Frame,
    hands: &[HandDetection],
    face: Option<&FaceDetection>,
    pose: Option<&PoseDetection>,
) -> Frame {
    let mut buf = match ImageBuffer::<Rgb<u8>, _>::from_raw(
        frame.width,
        frame.height,
        frame.data.clone(),
    ) {
        Some(b) => b,
        None => {
            // Dimension mismatch — return the input unchanged so the caller
            // still writes a JPEG (raw is also saved separately).
            return Frame::new(frame.data.clone(), frame.width, frame.height);
        }
    };

    for hand in hands {
        let color = match hand.side {
            Some(HandSide::Left) => Rgb([0, 255, 0]),
            Some(HandSide::Right) => Rgb([255, 102, 0]),
            None => Rgb([136, 136, 136]),
        };
        draw_hand(&mut buf, hand, color);
    }

    if let Some(f) = face {
        draw_face(&mut buf, f);
    }

    if let Some(p) = pose {
        draw_pose(&mut buf, p);
    }

    Frame::new(buf.into_raw(), frame.width, frame.height)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_frame() -> Frame {
        Frame::new(vec![128u8; 32 * 32 * 3], 32, 32)
    }

    #[test]
    fn no_landmarks_returns_same_size() {
        let frame = solid_frame();
        let result = annotate_frame(&frame, &[], None, None);
        assert_eq!(result.width, 32);
        assert_eq!(result.height, 32);
        assert_eq!(result.data.len(), 32 * 32 * 3);
    }

    #[test]
    fn drawing_hand_does_not_panic() {
        let mut landmarks = [Landmark { x: 0.0, y: 0.0, z: 0.0 }; 21];
        for (i, lm) in landmarks.iter_mut().enumerate() {
            lm.x = (i as f32) / 21.0;
            lm.y = 0.5;
        }
        let hand = HandDetection {
            side: None,
            landmarks,
            confidence: 1.0,
        };
        let frame = solid_frame();
        let result = annotate_frame(&frame, &[hand], None, None);
        assert_eq!(result.width, 32);
    }
}
