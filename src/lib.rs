mod detector;
mod models;
mod utils;

use crate::detector::{detect_face_from_image, SESSION};
use crate::models::FaceBox;
use crate::utils::get_path_from_cstr;
use ort::session::builder::GraphOptimizationLevel;
use ort::{ep, session::Session};
use std::os::raw::{c_char, c_int};
use std::time;

#[unsafe(no_mangle)]
pub extern "C" fn init_session(model_path: *const c_char) -> c_int {
    let model_path = match get_path_from_cstr(model_path) {
        Ok(path) => path,
        Err(err_code) => return err_code,
    };

    let session = match Session::builder() {
        Ok(builder) => {
            let mut builder = match builder.with_optimization_level(GraphOptimizationLevel::Level3)
            {
                Ok(builder) => match builder.with_intra_threads(num_cpus::get_physical()) {
                    Ok(builder) => {
                        match builder.with_execution_providers([ep::CUDA::default().build()]) {
                            Ok(builder) => builder,
                            Err(_) => return -3,
                        }
                    }
                    Err(_) => return -3,
                },
                Err(_) => return -3,
            };

            match builder.commit_from_file(model_path) {
                Ok(session) => session,
                Err(_) => return -4,
            }
        }
        Err(_) => return -5,
    };

    let mut global_session = SESSION.lock().unwrap();
    *global_session = Some(session);
    0
}

// #[unsafe(no_mangle)]
// pub extern "C" fn init_session(model_path: *const c_char) -> c_int {
//     let model_path = match get_path_from_cstr(model_path) {
//         Ok(path) => path,
//         Err(err_code) => return err_code,
//     };
//
//     let session = match Session::builder() {
//         Ok(builder) => {
//             let mut builder = match builder.with_execution_providers([ep::CUDA::default().build()]) {
//                 Ok(v) => v,
//                 Err(_) => return -3,
//             };
//
//             match builder.commit_from_file(model_path) {
//                 Ok(session) => session,
//                 Err(_) => return -4,
//             }
//         }
//         Err(_) => return -5,
//     };
//
//     let mut global_session = SESSION.lock().unwrap();
//     *global_session = Some(session);
//     0
// }

#[unsafe(no_mangle)]
pub extern "C" fn release_session() {
    let mut session = SESSION.lock().unwrap();
    *session = None;
}

#[unsafe(no_mangle)]
pub extern "C" fn detect_face_from_file(image_path: *const c_char, out_box: *mut FaceBox) -> c_int {
    let image_path = match get_path_from_cstr(image_path) {
        Ok(p) => p,
        Err(code) => return code,
    };

    let img = match image::open(image_path).map_err(|_| -3) {
        Ok(img) => img,
        Err(code) => return code,
    };

    // let mut tmp = DynamicImage::new_rgba8(img.width(), img.height());
    // img.clone_into(&mut tmp);
    // let images = vec![tmp; 16];

    let start = time::Instant::now();

    let face_box = match detect_face_from_image(img) {
        Ok(bb) => bb,
        Err(code) => return code,
    };

    // let outputs = detect_face_from_images(images);

    let elapsed = start.elapsed();
    println!("Processing time: {:?}", elapsed);

    // let mut face_box = Vec::new();
    // for res in outputs {
    //     face_box.push(res.unwrap_or_else(|_| FaceBox {
    //         x1: 0.0,
    //         y1: 0.0,
    //         x2: 0.0,
    //         y2: 0.0,
    //         confidence: 0.0,
    //     }));
    // }

    unsafe {
        *out_box = face_box;
    }

    // unsafe {
    //     for i in 0..face_box.len() {
    //         *out_box.add(i) = face_box[i];
    //     }
    // }

    0
}

#[unsafe(no_mangle)]
pub extern "C" fn detect_face_from_memory(
    bytes: *const u8,
    len: usize,
    out_box: *mut FaceBox,
) -> c_int {
    // 1. Input validation
    if bytes.is_null() || len == 0 {
        return -1; // invalid input
    }

    // 2. Build byte slice
    let byte_slice = unsafe { std::slice::from_raw_parts(bytes, len) };

    // 3. Decode image
    let img = match image::load_from_memory(byte_slice) {
        Ok(v) => v,
        Err(_) => return -4,
    };

    let face_box = match detect_face_from_image(img) {
        Ok(bb) => bb,
        Err(code) => return code,
    };

    unsafe {
        *out_box = face_box;
    }
    0
}
