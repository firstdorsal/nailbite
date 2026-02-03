//! Camera preview window with landmark overlays.
//!
//! Uses GTK `DrawingArea` + Cairo for rendering. Toggle-able via tray menu.
//! Shows the live camera feed with detected hand/face landmarks
//! drawn on top for debugging and tuning.

use std::cell::RefCell;
use std::f64::consts::PI;
use std::rc::Rc;

use gdk::prelude::GdkContextExt;
use gdk_pixbuf::{Colorspace, Pixbuf};
use glib::Propagation;
use gtk::prelude::*;

use crate::detection::types::{
    FaceDetection, FrameAnalysis, HandDetection, FINGERTIP_INDICES, OUTER_LIP_INDICES,
};

/// Data shared between the preview update path and the GTK draw callback.
struct PreviewData {
    pixbuf: Option<Pixbuf>,
    hands: Vec<HandDetection>,
    face: Option<FaceDetection>,
}

/// GTK-based camera preview window with landmark overlays.
pub struct PreviewWindow {
    window: gtk::Window,
    drawing_area: gtk::DrawingArea,
    data: Rc<RefCell<PreviewData>>,
}

impl Default for PreviewWindow {
    fn default() -> Self {
        Self::new()
    }
}

impl PreviewWindow {
    /// Create a new preview window. The window starts hidden.
    ///
    /// Must be called from the GTK main thread (after `gtk::init()`).
    pub fn new() -> Self {
        let window = gtk::Window::new(gtk::WindowType::Toplevel);
        window.set_title("Nailbite — Camera Preview");
        window.set_default_size(640, 480);
        window.set_resizable(true);

        let drawing_area = gtk::DrawingArea::new();
        drawing_area.set_size_request(640, 480);
        window.add(&drawing_area);

        // Hide on close instead of destroying, so it can be toggled back.
        window.connect_delete_event(|w, _| {
            w.hide();
            Propagation::Stop
        });

        let data = Rc::new(RefCell::new(PreviewData {
            pixbuf: None,
            hands: Vec::new(),
            face: None,
        }));

        // Connect the draw signal.
        let data_for_draw = Rc::clone(&data);
        drawing_area.connect_draw(move |widget, cr| {
            let data = data_for_draw.borrow();
            let alloc = widget.allocation();
            let widget_w = f64::from(alloc.width());
            let widget_h = f64::from(alloc.height());
            draw_preview(cr, &data, widget_w, widget_h);
            Propagation::Proceed
        });

        Self {
            window,
            drawing_area,
            data,
        }
    }

    /// Toggle the preview window visibility.
    pub fn toggle_visible(&self) {
        if self.window.is_visible() {
            self.window.hide();
        } else {
            self.window.show_all();
            self.window.present();
        }
    }

    /// Whether the preview window is currently visible.
    pub fn is_visible(&self) -> bool {
        self.window.is_visible()
    }

    /// Update the preview with new frame analysis data.
    ///
    /// Moves `raw_frame` out of the analysis to avoid copying pixel data.
    /// Only triggers a redraw if the preview window is visible.
    pub fn update(&self, analysis: &mut FrameAnalysis) {
        if !self.window.is_visible() {
            return;
        }

        let mut data = self.data.borrow_mut();

        if let Some(frame) = analysis.raw_frame.take() {
            if let (Ok(width), Ok(height)) =
                (i32::try_from(frame.width), i32::try_from(frame.height))
            {
                if width > 0 && height > 0 {
                    data.pixbuf = Some(Pixbuf::from_mut_slice(
                        frame.data,
                        Colorspace::Rgb,
                        false,
                        8,
                        width,
                        height,
                        width.saturating_mul(3),
                    ));
                }
            }
        }

        data.hands.clone_from(&analysis.hands);
        data.face.clone_from(&analysis.face);

        drop(data); // Release borrow before GTK call.
        self.drawing_area.queue_draw();
    }
}

/// Draw the preview frame and landmark overlays onto the Cairo context.
fn draw_preview(cr: &cairo::Context, data: &PreviewData, widget_w: f64, widget_h: f64) {
    // Black background.
    cr.set_source_rgb(0.0, 0.0, 0.0);
    let _ = cr.paint();

    let pixbuf = match &data.pixbuf {
        Some(pb) => pb,
        None => return,
    };

    let img_w = f64::from(pixbuf.width());
    let img_h = f64::from(pixbuf.height());
    if img_w <= 0.0 || img_h <= 0.0 {
        return;
    }

    // Scale to fit while maintaining aspect ratio.
    let scale_x = widget_w / img_w;
    let scale_y = widget_h / img_h;
    let scale = scale_x.min(scale_y);

    let offset_x = (widget_w - img_w * scale) / 2.0;
    let offset_y = (widget_h - img_h * scale) / 2.0;

    // Draw the camera frame.
    let _ = cr.save();
    cr.translate(offset_x, offset_y);
    cr.scale(scale, scale);
    cr.set_source_pixbuf(pixbuf, 0.0, 0.0);
    let _ = cr.paint();
    let _ = cr.restore();

    // Draw landmark overlays in widget coordinates.
    draw_hand_overlays(cr, &data.hands, img_w, img_h, scale, offset_x, offset_y);
    if let Some(face) = &data.face {
        draw_face_overlays(cr, face, img_w, img_h, scale, offset_x, offset_y);
    }
}

/// Convert a normalized landmark coordinate to widget pixel coordinates.
fn landmark_to_widget(
    lm_x: f64,
    lm_y: f64,
    img_w: f64,
    img_h: f64,
    scale: f64,
    offset_x: f64,
    offset_y: f64,
) -> (f64, f64) {
    (
        offset_x + lm_x * img_w * scale,
        offset_y + lm_y * img_h * scale,
    )
}

/// Hand skeleton connections (MediaPipe 21-landmark model).
const HAND_CONNECTIONS: &[(usize, usize)] = &[
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4), // Thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8), // Index
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12), // Middle
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16), // Ring
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20), // Pinky
    (5, 9),
    (9, 13),
    (13, 17), // Palm
];

/// Draw hand landmark overlays (skeleton + fingertip highlights).
#[allow(clippy::too_many_arguments)]
fn draw_hand_overlays(
    cr: &cairo::Context,
    hands: &[HandDetection],
    img_w: f64,
    img_h: f64,
    scale: f64,
    offset_x: f64,
    offset_y: f64,
) {
    for hand in hands {
        // Draw skeleton connections (green lines).
        cr.set_source_rgb(0.0, 0.78, 0.0);
        cr.set_line_width(1.5);
        for &(a, b) in HAND_CONNECTIONS {
            if let (Some(la), Some(lb)) = (hand.landmarks.get(a), hand.landmarks.get(b)) {
                let (x1, y1) = landmark_to_widget(
                    f64::from(la.x),
                    f64::from(la.y),
                    img_w,
                    img_h,
                    scale,
                    offset_x,
                    offset_y,
                );
                let (x2, y2) = landmark_to_widget(
                    f64::from(lb.x),
                    f64::from(lb.y),
                    img_w,
                    img_h,
                    scale,
                    offset_x,
                    offset_y,
                );
                cr.move_to(x1, y1);
                cr.line_to(x2, y2);
            }
        }
        let _ = cr.stroke();

        // Draw all landmark points (small green circles).
        cr.set_source_rgb(0.0, 1.0, 0.0);
        for lm in &hand.landmarks {
            let (x, y) = landmark_to_widget(
                f64::from(lm.x),
                f64::from(lm.y),
                img_w,
                img_h,
                scale,
                offset_x,
                offset_y,
            );
            cr.arc(x, y, 3.0, 0.0, 2.0 * PI);
            let _ = cr.fill();
        }

        // Highlight fingertips (larger red circles).
        cr.set_source_rgb(1.0, 0.39, 0.39);
        for &idx in &FINGERTIP_INDICES {
            if let Some(lm) = hand.landmarks.get(idx) {
                let (x, y) = landmark_to_widget(
                    f64::from(lm.x),
                    f64::from(lm.y),
                    img_w,
                    img_h,
                    scale,
                    offset_x,
                    offset_y,
                );
                cr.arc(x, y, 5.0, 0.0, 2.0 * PI);
                let _ = cr.fill();
            }
        }
    }
}

/// Draw face landmark overlays (outer lip contour).
#[allow(clippy::too_many_arguments)]
fn draw_face_overlays(
    cr: &cairo::Context,
    face: &FaceDetection,
    img_w: f64,
    img_h: f64,
    scale: f64,
    offset_x: f64,
    offset_y: f64,
) {
    let lip_points: Vec<(f64, f64)> = OUTER_LIP_INDICES
        .iter()
        .filter_map(|&idx| face.landmarks.get(idx))
        .map(|lm| {
            landmark_to_widget(
                f64::from(lm.x),
                f64::from(lm.y),
                img_w,
                img_h,
                scale,
                offset_x,
                offset_y,
            )
        })
        .collect();

    if lip_points.len() > 1 {
        cr.set_source_rgb(1.0, 0.78, 0.0); // Yellow
        cr.set_line_width(1.5);

        if let Some(&(start_x, start_y)) = lip_points.first() {
            cr.move_to(start_x, start_y);
            for &(x, y) in lip_points.get(1..).unwrap_or(&[]) {
                cr.line_to(x, y);
            }
            cr.close_path();
            let _ = cr.stroke();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn landmark_to_widget_maps_correctly() {
        // Normalized (0.5, 0.5) on a 640x480 image at 1:1 scale, no offset.
        let (x, y) = landmark_to_widget(0.5, 0.5, 640.0, 480.0, 1.0, 0.0, 0.0);
        assert!((x - 320.0).abs() < 0.001);
        assert!((y - 240.0).abs() < 0.001);
    }

    #[test]
    fn landmark_to_widget_with_scale_and_offset() {
        let (x, y) = landmark_to_widget(0.5, 0.5, 640.0, 480.0, 2.0, 10.0, 20.0);
        let expected_x = 10.0 + 0.5 * 640.0 * 2.0;
        let expected_y = 20.0 + 0.5 * 480.0 * 2.0;
        assert!((x - expected_x).abs() < 0.001);
        assert!((y - expected_y).abs() < 0.001);
    }

    #[test]
    fn landmark_to_widget_origin() {
        let (x, y) = landmark_to_widget(0.0, 0.0, 640.0, 480.0, 1.0, 10.0, 20.0);
        assert!((x - 10.0).abs() < 0.001);
        assert!((y - 20.0).abs() < 0.001);
    }

    #[test]
    fn landmark_to_widget_bottom_right() {
        let (x, y) = landmark_to_widget(1.0, 1.0, 640.0, 480.0, 1.0, 0.0, 0.0);
        assert!((x - 640.0).abs() < 0.001);
        assert!((y - 480.0).abs() < 0.001);
    }

    #[test]
    fn hand_connections_are_valid_indices() {
        for &(a, b) in HAND_CONNECTIONS {
            assert!(a < 21, "Connection start {a} out of range for 21 landmarks");
            assert!(b < 21, "Connection end {b} out of range for 21 landmarks");
        }
    }
}
