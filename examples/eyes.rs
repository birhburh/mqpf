use {
    macroquad::{
        miniquad::{
            conf::{AppleGfxApi, Platform},
            window::{dpi_scale, screen_size},
        },
        prelude::*,
    },
    mqpf::{push_path, ArcDirection, Path2D, Renderer, Scene, PI_2},
    std::f32::consts::PI,
};

fn window_conf() -> Conf {
    let apple_gfx_api = AppleGfxApi::OpenGl;
    let high_dpi = true;
    let window_width = 600;
    let window_height = window_width * 3 / 4;
    Conf {
        window_title: format!("Face").to_owned(),
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
    let eyes_rect = Rect::new(
        framebuffer_size.0 as f32 / hidpi_factor as f32 / 2.
            - ((framebuffer_size.0 * 0.9) / 4.0) as f32,
        framebuffer_size.1 as f32 / hidpi_factor as f32 / 2.
            - ((framebuffer_size.1 * 0.9) / 4.0) as f32,
        ((framebuffer_size.0 * 0.9) / 2.0) as f32,
        ((framebuffer_size.1 * 0.9) / 2.0) as f32,
    );
    let eyes_radii = eyes_rect.size() * vec2(0.23, 0.5);
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
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut framebuffer_size = screen_size();
    let hidpi_factor = dpi_scale();

    let context = unsafe { get_internal_gl().quad_context };
    let mut renderer = Renderer::new(context, framebuffer_size, color_u8!(77, 77, 82, 255));

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

        let mut path = Path2D::new();
        let mouth_pos = vec2(
            framebuffer_size.0 / hidpi_factor as f32,
            framebuffer_size.1 / hidpi_factor as f32,
        ) * vec2(0.5, 0.85);
        path.arc(mouth_pos, mouth_pos.x * 0.18, 0.0, PI, ArcDirection::CW);
        path.arc(mouth_pos, mouth_pos.x * 0.16, PI, 0.0, ArcDirection::CCW);
        path.close_path();
        push_path(
            &mut canvas_scene,
            &transform,
            path,
            &color_u8!(220, 110, 110, 255),
        );

        renderer.render(canvas_scene);

        // break;
        next_frame().await;
    }
}
