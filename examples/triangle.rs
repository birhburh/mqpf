use {
    macroquad::{
        miniquad::{
            conf::{AppleGfxApi, Platform},
            window::screen_size,
        },
        prelude::*,
        ui::hash,
    },
    mqpf::Renderer,
    pathfinder_geometry::{transform2d::Transform2F, vector::vec2f},
};

fn window_conf() -> Conf {
    let apple_gfx_api = AppleGfxApi::OpenGl;
    let high_dpi = true;
    let window_width = 600;
    let window_height = window_width * 3 / 4;
    Conf {
        window_title: format!("TRIANGLE").to_owned(),
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

    let start_time = get_time();
    loop {
        clear_background(DARKGRAY);

        if screen_width() != saved_width || screen_height() != saved_height {
            saved_width = screen_width();
            saved_height = screen_height();

            framebuffer_size = screen_size();
            renderer.update_viewport(framebuffer_size);
        }

        let elapsed_time = get_time() - start_time;

        let path = renderer.begin_path(hash!());

        path.move_to(vec2f(30.0, 30.0));
        path.line_to(vec2f(100.0, 100.0));
        // path.line_to(vec2f(100.0, 30.0));
        if elapsed_time < 5.0 {
            path.line_to(vec2f(100.0 + elapsed_time as f32 * 10.0, 30.0));
        }
        else {
            path.line_to(vec2f(150.0, 30.0));
        }

        // path.move_to(vec2f(250.0, 30.0));
        // path.quadratic_curve_to(vec2f(330.0, 30.0), vec2f(330.0, 80.0));

        let path_id = path.id;
        renderer.fill_path(
            &Transform2F::default(),
            // &Transform2F::from_translation(vec2f(30.0, 30.0)),
            // &Transform2F::from_translation(vec2f(0.0, elapsed_time as f32 * 20.0)),
            path_id,
            &color_u8!(220, 220, 220, 255),
        );

        let path = renderer.begin_path(hash!());

        path.move_to(vec2f(300.0, 300.0));
        path.line_to(vec2f(360.0, 360.0));
        path.line_to(vec2f(370.0, 300.0));

        let path_id = path.id;
        renderer.fill_path(
            &Transform2F::default(),
            path_id,
            &color_u8!(110, 110, 12, 255),
        );

        renderer.render();

        dbg!(elapsed_time);
        if elapsed_time > 0.25 {
        // if elapsed_time > 5.0 {
            break;
        }
        next_frame().await;
    }
}
