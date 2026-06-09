mod configs;
mod detector;
mod errors;
mod models;
mod utils;

use crate::detector::{SESSION, detect_face};
use crate::errors::FaceDetectionResponse;
use crate::models::FaceBox;
use crate::utils::get_path_from_cstr;
use ort::session::builder::GraphOptimizationLevel;
use ort::{ep, session::Session};
use std::os::raw::{c_char, c_int};
// use std::time;

#[unsafe(no_mangle)]
pub extern "C" fn init_session(model_path: *const c_char) -> c_int {
    let model_path = match get_path_from_cstr(model_path) {
        Ok(path) => path,
        Err(code) => return code,
    };

    if !model_path.is_file() {
        return FaceDetectionResponse::InvalidONNXModelPath as c_int;
    }

    let session = match Session::builder() {
        Ok(builder) => {
            let mut builder = match builder.with_optimization_level(GraphOptimizationLevel::Level3)
            {
                Ok(builder) => match builder.with_intra_threads(num_cpus::get_physical()) {
                    Ok(builder) => {
                        match builder.with_execution_providers([ep::CUDA::default().build()]) {
                            Ok(builder) => builder,
                            Err(_) => {
                                return FaceDetectionResponse::ExecutionProviderError as c_int;
                            }
                        }
                    }
                    Err(_) => return FaceDetectionResponse::SessionInitializationError as c_int,
                },
                Err(_) => return FaceDetectionResponse::SessionInitializationError as c_int,
            };

            match builder.commit_from_file(model_path) {
                Ok(session) => session,
                Err(_) => return FaceDetectionResponse::ONNXModelLoadingError as c_int,
            }
        }
        Err(_) => return FaceDetectionResponse::SessionBuilderError as c_int,
    };

    let mut global_session = SESSION.lock().unwrap();
    *global_session = Some(session);

    FaceDetectionResponse::Success as c_int
}

#[unsafe(no_mangle)]
pub extern "C" fn release_session() {
    let mut session = SESSION.lock().unwrap();
    *session = None;
}

#[unsafe(no_mangle)]
pub extern "C" fn detect_face_from_file(image_path: *const c_char, out_box: *mut FaceBox) -> c_int {
    let image_path = match get_path_from_cstr(image_path) {
        Ok(path) => path,
        Err(code) => return code,
    };

    if !image_path.is_file() {
        return FaceDetectionResponse::InvalidImagePath as c_int;
    }

    let img = match image::open(image_path)
        .map_err(|_| FaceDetectionResponse::ImageLoadingError as c_int)
    {
        Ok(img) => img,
        Err(code) => return code,
    };

    // let start = time::Instant::now();
    let face_box = match detect_face(img) {
        Ok(bb) => bb,
        Err(code) => return code as c_int,
    };
    // let elapsed = start.elapsed();
    // println!("Processing time: {:?}", elapsed);

    unsafe {
        *out_box = face_box;
    }
    FaceDetectionResponse::Success as c_int
}

#[unsafe(no_mangle)]
pub extern "C" fn detect_face_from_memory(
    bytes: *const u8,
    len: usize,
    out_box: *mut FaceBox,
) -> c_int {
    if bytes.is_null() || len == 0 {
        return FaceDetectionResponse::EmptyInputImage as c_int;
    }

    let byte_slice = unsafe { std::slice::from_raw_parts(bytes, len) };

    let img = match image::load_from_memory(byte_slice) {
        Ok(v) => v,
        Err(_) => return FaceDetectionResponse::ImageLoadingError as c_int,
    };

    let face_box = match detect_face(img) {
        Ok(bb) => bb,
        Err(code) => return code,
    };

    unsafe {
        *out_box = face_box;
    }
    FaceDetectionResponse::Success as c_int
}

// pub fn detect_face_from_batch {}
// let mut tmp = DynamicImage::new_rgba8(img.width(), img.height());
// img.clone_into(&mut tmp);
// let images = vec![tmp; 16];

// let outputs = detect_face_from_images(images);

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

// unsafe {
//     for i in 0..face_box.len() {
//         *out_box.add(i) = face_box[i];
//     }
// }
