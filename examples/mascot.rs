use macroquad::{
    miniquad::{
        conf::{AppleGfxApi, Platform},
        window::{dpi_scale, screen_size},
    },
    prelude::*,
};
use mqpf::{push_path, Path2D, Renderer, Scene, PI_2};

fn window_conf() -> Conf {
    let apple_gfx_api = AppleGfxApi::OpenGl;
    let high_dpi = true;
    let window_width = 600;
    let window_height = window_width * 3 / 4;
    Conf {
        window_title: format!("OUR BELOVED MASCOT").to_owned(),
        platform: Platform {
            apple_gfx_api,
            // blocking_event_loop: true,
            ..Default::default()
        },
        window_width,
        window_height,
        high_dpi,
        ..Default::default()
    }
}

fn draw_eyes(
    scene: &mut Scene,
    transform: &Affine2,
    framebuffer_size: (f32, f32),
    hidpi_factor: f32,
    mouse_position: (f32, f32),
    time: f64,
) {
    // let time: f64 = 0.0;
    // let mouse_position = Vector2F::new(0.0, 0.0);
    let eyes_rect = Rect::new(
        framebuffer_size.0 as f32 / hidpi_factor as f32 / 2.
            - ((framebuffer_size.0 * 0.9) / 4.0) as f32,
        framebuffer_size.1 as f32 / hidpi_factor as f32 / 2.
            - ((framebuffer_size.1 * 0.9) / 4.0) as f32,
        ((framebuffer_size.0 * 0.9) / 2.0) as f32,
        ((framebuffer_size.1 * 0.9) / 2.0) as f32,
    );
    let eyes_radii = eyes_rect.size() * vec2(0.23, 0.35);
    let eyes_left_position = eyes_rect.point() + eyes_radii;
    let eyes_right_position = eyes_rect.point() + vec2(eyes_rect.w - eyes_radii.x, eyes_radii.y);
    let eyes_center = f32::min(eyes_radii.x, eyes_radii.y) * 0.5;
    let blink = (1.0 - f64::powf((time * 0.5).sin(), 200.0) * 0.8) as f32;

    let mut path = Path2D::new();
    path.ellipse(eyes_left_position, eyes_radii, 0.0, 0.0, PI_2);
    path.ellipse(eyes_right_position, eyes_radii, 0.0, 0.0, PI_2);
    push_path(scene, transform, path, &color_u8!(220, 220, 220, 255));

    let mut delta = (vec2(mouse_position.0, mouse_position.1) - eyes_right_position) / (eyes_radii);
    let distance = delta.length();
    if distance > 1.0 {
        delta *= 1.0 / distance;
    }
    delta *= eyes_radii * vec2(0.4, 0.5);
    let mut path = Path2D::new();
    path.ellipse(
        eyes_left_position + delta + vec2(0.0, eyes_radii.y * 0.25 * (1.0 - blink)),
        vec2(eyes_center, eyes_center * blink),
        0.0,
        0.0,
        PI_2,
    );
    path.ellipse(
        eyes_right_position + delta + vec2(0.0, eyes_radii.y * 0.25 * (1.0 - blink)),
        vec2(eyes_center, eyes_center * blink),
        0.0,
        0.0,
        PI_2,
    );
    push_path(scene, transform, path, &color_u8!(32, 32, 32, 255));

    let mut path = Path2D::new();
    path.ellipse(
        eyes_left_position
            + delta
            + vec2(
                eyes_center * 0.5,
                eyes_radii.y * 0.25 * (1.0 - blink) + eyes_center * 0.75 * (blink - 0.5),
            ),
        vec2(eyes_center * 0.2, eyes_center * 0.25),
        0.0,
        0.0,
        PI_2,
    );
    path.ellipse(
        eyes_right_position
            + delta
            + vec2(
                eyes_center * 0.5,
                eyes_radii.y * 0.25 * (1.0 - blink) + eyes_center * 0.75 * (blink - 0.5),
            ),
        vec2(eyes_center * 0.2, eyes_center * 0.25),
        0.0,
        0.0,
        PI_2,
    );
    push_path(scene, transform, path, &color_u8!(220, 220, 220, 255));

    let mut path = Path2D::new();
    let tooth_pos = vec2(
        framebuffer_size.0 / hidpi_factor as f32,
        framebuffer_size.1 / hidpi_factor as f32,
    ) * vec2(0.45, 0.83);
    path.move_to(tooth_pos);
    path.line_to(tooth_pos * vec2(1.0, 1.1));
    path.bezier_curve_to(
        tooth_pos * vec2(1.0, 1.15),
        tooth_pos * vec2(1.1, 1.15),
        tooth_pos * vec2(1.1, 1.1),
    );
    path.line_to(tooth_pos * vec2(1.1, 1.0));
    path.close_path();
    push_path(scene, transform, path, &color_u8!(220, 220, 220, 255));

    let mut path = Path2D::new();
    let tooth_pos = vec2(
        framebuffer_size.0 / hidpi_factor as f32,
        framebuffer_size.1 / hidpi_factor as f32,
    ) * vec2(0.5, 0.83);
    path.move_to(tooth_pos);
    path.line_to(tooth_pos * vec2(1.0, 1.1));
    path.bezier_curve_to(
        tooth_pos * vec2(1.0, 1.15),
        tooth_pos * vec2(1.1, 1.15),
        tooth_pos * vec2(1.1, 1.1),
    );
    path.line_to(tooth_pos * vec2(1.1, 1.0));
    path.close_path();
    push_path(scene, transform, path, &color_u8!(220, 220, 220, 255));

    let mut path = Path2D::new();
    let mouth_pos = vec2(
        framebuffer_size.0 / hidpi_factor as f32,
        framebuffer_size.1 / hidpi_factor as f32,
    ) * vec2(0.68, 0.75);
    path.move_to(mouth_pos);
    path.bezier_curve_to(
        mouth_pos * vec2(0.85, 0.85),
        mouth_pos * vec2(0.65, 0.85),
        mouth_pos * vec2(0.5, 1.0),
    );
    path.bezier_curve_to(
        mouth_pos * vec2(0.43, 1.08),
        mouth_pos * vec2(0.43, 1.16),
        mouth_pos * vec2(0.5, 1.2),
    );
    path.bezier_curve_to(
        mouth_pos * vec2(0.5, 1.21),
        mouth_pos * vec2(0.6, 1.21),
        mouth_pos * vec2(0.65, 1.15),
    );
    path.bezier_curve_to(
        mouth_pos * vec2(0.65, 1.15),
        mouth_pos * vec2(0.75, 1.07),
        mouth_pos * vec2(0.85, 1.15),
    );
    path.bezier_curve_to(
        mouth_pos * vec2(0.9, 1.17),
        mouth_pos * vec2(0.95, 1.2),
        mouth_pos * vec2(1.0, 1.2),
    );
    path.bezier_curve_to(
        mouth_pos * vec2(1.09, 1.15),
        mouth_pos * vec2(1.05, 1.05),
        mouth_pos * vec2(1.0, 1.0),
    );
    path.close_path();
    push_path(scene, transform, path, &color_u8!(246, 210, 165, 255));

    let mut path = Path2D::new();
    let nose_pos = vec2(
        framebuffer_size.0 / hidpi_factor as f32,
        framebuffer_size.1 / hidpi_factor as f32,
    ) * vec2(0.5, 0.7);
    path.ellipse(
        nose_pos,
        vec2(eyes_center, eyes_center * 0.7),
        0.0,
        0.0,
        PI_2,
    );
    push_path(scene, transform, path, &color_u8!(32, 32, 32, 255));
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut framebuffer_size = screen_size();
    let hidpi_factor = dpi_scale();

    let context = unsafe { get_internal_gl().quad_context };
    let mut renderer = Renderer::new(context, framebuffer_size, color_u8!(116, 200, 214, 255));

    let mut saved_width = 0.0;
    let mut saved_height = 0.0;

    let start_time = get_time();
    loop {
        clear_background(DARKGRAY);

        if screen_width() != saved_width || screen_height() != saved_height {
            saved_width = screen_width();
            saved_height = screen_height();

            framebuffer_size = screen_size();
            renderer.update_viewport(framebuffer_size);
        }

        let cursor_position = mouse_position();

        let mut canvas_scene = Scene {
            view_box: Rect::new(0.0, 0.0, framebuffer_size.0, framebuffer_size.1),
            ..Default::default()
        };
        let transform = Affine2::from_scale(vec2(hidpi_factor, hidpi_factor));

        let frame_start_time = get_time();
        let frame_start_elapsed_time = frame_start_time - start_time;
        draw_eyes(
            &mut canvas_scene,
            &transform,
            framebuffer_size,
            hidpi_factor,
            cursor_position,
            frame_start_elapsed_time,
        );
        renderer.render(canvas_scene);

        next_frame().await;
    }
}
