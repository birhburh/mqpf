use {
    macroquad::{
        miniquad::{
            conf::{AppleGfxApi, Platform},
            window::screen_size,
        },
        prelude::*,
    },
    mqpf::{push_path, Path2D, Renderer, Scene, PI_2},
    pathfinder_geometry::{
        rect::RectF,
        transform2d::Transform2F,
        vector::{vec2f, Vector2F},
    },
};

fn window_conf() -> Conf {
    let apple_gfx_api = AppleGfxApi::OpenGl;
    let high_dpi = true;
    let window_width = 600;
    let window_height = window_width * 3 / 4;
    Conf {
        window_title: format!("FIXED EYES").to_owned(),
        platform: Platform {
            apple_gfx_api,
            ..Default::default()
        },
        window_width,
        window_height,
        high_dpi,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut framebuffer_size = screen_size();

    let context = unsafe { get_internal_gl().quad_context };
    let mut renderer = Renderer::new(context, framebuffer_size, color_u8!(77, 77, 82, 255));

    let mut saved_width = 0.0;
    let mut saved_height = 0.0;

    loop {
        clear_background(DARKGRAY);

        if screen_width() != saved_width || screen_height() != saved_height {
            saved_width = screen_width();
            saved_height = screen_height();

            framebuffer_size = screen_size();
            renderer.update_viewport(framebuffer_size);
        }

        let mut canvas_scene = Scene {
            view_box: RectF::new(
                Vector2F::zero(),
                Vector2F::new(framebuffer_size.0, framebuffer_size.1),
            ),
            ..Default::default()
        };

        let mut path = Path2D::new();
        path.ellipse(vec2f(180.0, 250.0), vec2f(160.0, 230.0), 0.0, 0.0, PI_2);
        push_path(
            &mut canvas_scene,
            &Transform2F::default(),
            path,
            &color_u8!(220, 220, 220, 255),
        );

        renderer.render(&canvas_scene);

        next_frame().await;
    }
}
