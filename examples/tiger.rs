use {
    macroquad::{
        miniquad::{
            conf::{AppleGfxApi, Platform},
            window::screen_size,
        },
        prelude::*,
    },
    mqpf::{svg::render_nodes, Renderer, Scene},
    pathfinder_geometry::{rect::RectF, transform2d::Transform2F, vector::vec2f},
    usvg::Tree as SvgTree,
};

fn window_conf() -> Conf {
    let apple_gfx_api = AppleGfxApi::OpenGl;
    let high_dpi = true;
    let window_width = 600;
    let window_height = window_width * 3 / 4;
    Conf {
        window_title: format!("Tiger").to_owned(),
        platform: Platform {
            apple_gfx_api,
            // blocking_event_loop: true,
            ..Default::default()
        },
        // fullscreen: true,
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

    let svg_data = include_bytes!("../svgs/Ghostscript_Tiger.svg");
    let tree = SvgTree::from_data(svg_data, &usvg::Options::default()).unwrap();

    let mut canvas_scene = Scene {
        view_box: RectF::new(
            vec2f(0.0, 0.0),
            vec2f(framebuffer_size.0, framebuffer_size.1),
        ),
        ..Default::default()
    };

    let start_time = get_time();
    loop {
        clear_background(DARKGRAY);

        if screen_width() != saved_width || screen_height() != saved_height {
            saved_width = screen_width();
            saved_height = screen_height();

            framebuffer_size = screen_size();
            renderer.update_viewport(framebuffer_size);

            canvas_scene = Scene {
                view_box: RectF::new(
                    vec2f(0.0, 0.0),
                    vec2f(framebuffer_size.0, framebuffer_size.1),
                ),
                ..Default::default()
            };

            let side_size = tree.size().width().min(tree.size().height());
            let scale = if screen_width() < screen_height() {
                framebuffer_size.0 / side_size * 0.9
            } else {
                framebuffer_size.1 / side_size * 0.9
            };

            let mut transform = Transform2F::from_translation(vec2f(
                framebuffer_size.0 / 2.0 - side_size * scale / 2.0,
                framebuffer_size.1 / 2.0 - side_size * scale / 2.0,
            ));
            transform *= Transform2F::from_scale(vec2f(scale, scale));

            render_nodes(&tree.root(), &mut canvas_scene, transform);
        }

        renderer.render(&canvas_scene);

        if get_time() - start_time > 20.0 {
            break;
        }
        next_frame().await;
    }
}
