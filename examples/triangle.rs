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

use macroquad::miniquad::native::android::{
    self,
    ndk_sys::{self},
    ndk_utils,
};
use once_cell::sync::Lazy;
use std::ffi::c_int;
use std::ffi::CString;
use std::panic;
use std::str;
use std::sync::{Arc, Mutex};

struct GlobalData {
    openfile: ndk_sys::jobject,
    data: Option<Arc<Mutex<Option<String>>>>,
    finish: Option<Arc<Mutex<bool>>>,
}
unsafe impl Send for GlobalData {}
unsafe impl Sync for GlobalData {}
static GLOBALS: Lazy<Mutex<GlobalData>> = Lazy::new(|| {
    Mutex::new(GlobalData {
        openfile: std::ptr::null_mut(),
        data: None,
        finish: None,
    })
});
#[no_mangle]
pub unsafe extern "C" fn Java_rust_mqpf_example_triangle_FileOpen_init() {
    let env = android::attach_jni_env();
    let mut globals = GLOBALS.lock().unwrap();
    let openfile = ndk_utils::new_object!(env, "rust/mqpf/example/triangle/FileOpen", "()V");
    assert!(!openfile.is_null());
    globals.openfile = ndk_utils::new_global_ref!(env, openfile);
}
#[no_mangle]
pub unsafe extern "C" fn Java_rust_mqpf_example_triangle_FileOpen_saveUri(
    env: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
    array: ndk_sys::jbyteArray,
) {
    let mut globals = GLOBALS.lock().unwrap();
    let len = ((**env).GetArrayLength.unwrap())(env, array);
    let elements = ((**env).GetByteArrayElements.unwrap())(env, array, std::ptr::null_mut());
    let data = std::slice::from_raw_parts(elements as *mut u8, len as usize);
    if let Some(ref mut d) = globals.data {
        let s = match str::from_utf8(data) {
            Ok(v) => v.to_string(),
            Err(e) => panic!("Invalid UTF-8 sequence: {}", e),
        };
        *d.lock().unwrap() = Some(s);
    }
    ((**env).ReleaseByteArrayElements.unwrap())(env, array, elements, 0);
}
#[no_mangle]
pub unsafe extern "C" fn Java_rust_mqpf_example_triangle_FileOpen_finish(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
    _: ndk_sys::jbyteArray,
) {
    let mut globals = GLOBALS.lock().unwrap();
    if let Some(ref mut f) = globals.finish {
        *f.lock().unwrap() = false;
    }
}
fn finish_main_activity() {
    let env = unsafe { android::attach_jni_env() };
    let globals = GLOBALS.lock().unwrap();
    unsafe {
        ndk_utils::call_void_method!(env, globals.openfile, "finishMainActivity", "()V");
    }
}

fn find_file(data: Arc<Mutex<Option<String>>>, finish: Arc<Mutex<bool>>) {
    let env = unsafe { android::attach_jni_env() };
    let openfile;
    {
        let mut globals = GLOBALS.lock().unwrap();
        globals.data = Some(data);
        globals.finish = Some(finish);
        openfile = globals.openfile;
    }
    unsafe {
        ndk_utils::call_void_method!(env, openfile, "OpenFileDialog", "()V");
    }
}

fn log_this_int(s: &str, first: bool) {
    let env = unsafe { android::attach_jni_env() };
    let mut globals = GLOBALS.lock().unwrap();
    unsafe {
        let c_string = CString::new(s).expect("CString conversion failed");
        let java_arg = (**env).NewStringUTF.unwrap()(env, c_string.as_ptr());
        let r = panic::catch_unwind(|| {
            ndk_utils::call_void_method!(
                env,
                globals.openfile,
                "logThis",
                "(Ljava/lang/String;Z)V",
                java_arg,
                first as c_int
            );
        });
        if let Some(ref mut d) = globals.data {
            let s = match r {
                Ok(_) => "No panic".to_string(),
                Err(e) => {
                    if let Some(msg) = e.downcast_ref::<&str>() {
                        format!("Panic occurred: {}", msg)
                    } else if let Some(msg) = e.downcast_ref::<String>() {
                        format!("Panic occurred: {}", msg)
                    } else {
                        "Unknown panic occurred".to_string()
                    }
                }
            };
            *d.lock().unwrap() = Some(s);
        }
        (**env).DeleteLocalRef.unwrap()(env, java_arg as *mut _);
    }
}

fn log_first(text: &str) {
    log_this_int(text, true);
}

fn log_this(text: &str) {
    log_this_int(text, false);
}

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
    let mut exit = false;
    let data = Arc::new(Mutex::new(None));
    let finish = Arc::new(Mutex::new(false));
    let mut first = true;

    loop {
        let ref mut val = *finish.lock().unwrap();
        // not call new find_file until
        // current is not closed
        if *val == false {
            // if find_file returned and
            // there is no data set
            // close activity
            if !first {
                exit = true;
                break;
            }
            *val = true;
            first = false;
            find_file(data.clone(), finish.clone());
        }
        if let Some(_) = *data.lock().unwrap() {
            break;
        }
    }

    // let mut text0 = "".to_string();
    log_first("I AM STARTING, MR KRABS!\n");
    log_this("I AM C...\n");
    log_this("ONTINUE!\n");

    let mut framebuffer_size = screen_size();

    let context = unsafe { get_internal_gl().quad_context };
    let mut renderer = Renderer::new(context, framebuffer_size);

    let mut saved_width = 0.0;
    let mut saved_height = 0.0;

    let start_time = get_time();
    loop {
        clear_background(WHITE);
        // clear_background(DARKGRAY);
        if exit {
            finish_main_activity();
        }
        // {
        //     let ref mut v = *data.lock().unwrap();
        //     if let Some(v) = v {
        //         text0 = v.clone();
        //     }
        //     *v = None;
        // }
        // draw_text(&text0, 10., 10., 20., BLACK);

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
        if elapsed_time < 10.0 {
            path.line_to(vec2f(100.0 + elapsed_time as f32 * 30.0, 30.0));
            // path.line_to(vec2f(150.0 - elapsed_time as f32 * 10.0, 30.0));
        } else {
            path.line_to(vec2f(400.0, 30.0));
            // path.line_to(vec2f(100.0, 30.0));
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

        let path = renderer.begin_path(hash!());

        path.move_to(vec2f(350.0, 30.0));
        path.quadratic_curve_to(vec2f(430.0, 30.0), vec2f(430.0, 80.0));

        let path_id = path.id;
        renderer.fill_path(
            &Transform2F::default(),
            path_id,
            &color_u8!(180, 255, 180, 255),
        );

        let path = renderer.begin_path(hash!());

        path.move_to(vec2f(365.0, 30.0));
        path.quadratic_curve_to(vec2f(445.0, 30.0), vec2f(445.0, 80.0));

        let path_id = path.id;
        renderer.fill_path(
            &Transform2F::default(),
            path_id,
            &color_u8!(180, 128, 0, 128),
        );

        renderer.render();

        dbg!(elapsed_time);
        // if elapsed_time > 0.25 {
        if elapsed_time > 6.0 {
            // break;
        }
        next_frame().await;
    }
}
