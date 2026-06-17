//! C Foreign Function Interface (FFI) bindings for face detection functionality
//!
//! This module provides C-compatible functions for initializing the face detection model,
//! releasing session resources, and detecting faces from image files or memory buffers.
//! These bindings allow the Rust face detection library to be called from C, C++, and
//! other languages with C FFI support.
//!
//! # Organization
//! - **Session Management**: [`init_session`], [`release_session`] - Initialize and clean up the ONNX session
//! - **Face Detection**: [`detect_face_from_file`], [`detect_face_from_memory`] - Detect faces from different input sources
//!
//! # Safety
//! All exported functions are `unsafe extern "C"` and must be called correctly from C code.
//! Callers must provide valid pointers and handle null pointer checks appropriately.
//!
//! # Example (C code)
//! ```c
//! #include "facedetector.h"
//!
//! int main() {
//!     // Initialize session
//!     init_session("model.onnx");
//!     
//!     // Detect face
//!     FaceBox box;
//!     int result = detect_face_from_file("face.jpg", &box);
//!     if (result == 0) {
//!         printf("Face at: x1=%f, y1=%f, x2=%f, y2=%f\n",
//!                box.x1, box.y1, box.x2, box.y2);
//!     }
//!     
//!     // Clean up
//!     release_session();
//!     return 0;
//! }
//! ```

mod configs;
mod detector;
mod models;
mod utils;

use crate::detector::{SESSION, detect_face, init_onnx_session};
use crate::models::{FaceBox, FaceDetectionResponse};
use crate::utils::convert_cstring_to_path;

use std::os::raw::{c_char, c_int};
// use std::time;

/// Initializes the face detection ONNX session from a model file path.
///
/// This function must be called before any detection functions. It loads the model,
/// configures optimization settings, and initializes the CUDA backend if available.
/// The session is stored in a global static variable and reused across calls.
///
/// # Arguments
/// * `model_path` - C string pointer to the ONNX model file path (UTF-8 encoded)
///
/// # Returns
/// * `0` - Success (corresponds to `FaceDetectionResponse::Success`)
/// * Non-zero error code from `FaceDetectionResponse` enum indicating failure:
///   - `InvalidONNXModelPath` - File does not exist or path is invalid
///   - `ExecutionProviderError` - CUDA execution provider setup failed
///   - `SessionInitializationError` - Failed to configure optimization or threads
///   - `ONNXModelLoadingError` - Failed to load model from file
///   - `SessionBuilderError` - Failed to create session builder
///
/// # Safety
/// This function is `unsafe extern "C"` and must be called with a valid pointer:
/// - `model_path` must be a valid null-terminated UTF-8 C string
/// - The pointer must be non-null (checked by `convert_cstring_to_path`)
/// - The string data must remain valid for the duration of the call
///
/// # Panics
/// May panic if the global session mutex is poisoned or if UTF-8 conversion fails
/// (though invalid UTF-8 will return an error code instead of panicking).
///
/// # Examples
/// ```rust,no_run
/// use std::ffi::CString;
/// # use your_crate::ffi::init_session;
///
/// let path = CString::new("models/face_detector.onnx").unwrap();
/// let result = unsafe { init_session(path.as_ptr()) };
/// assert_eq!(result, 0);
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn init_session(model_path: *const c_char) -> c_int {
    let model_path = match convert_cstring_to_path(model_path) {
        Ok(path) => path,
        Err(code) => return code,
    };
    init_onnx_session(model_path) as c_int
}

/// Releases the global ONNX session and frees associated resources.
///
/// This function should be called when face detection is no longer needed to properly
/// clean up GPU memory, CUDA contexts, and other session resources. After calling
/// `release_session`, [`init_session`] must be called again before any detection.
///
/// # Safety
/// This function is safe to call even if the session was not initialized or already released.
/// However, it modifies global state and should not be called while other threads are
/// performing detection operations without external synchronization.
///
/// # Panics
/// May panic if the global session mutex is poisoned.
///
/// # Examples
/// ```rust,no_run
/// # use your_crate::ffi::release_session;
/// // Clean up after all detection is complete
/// unsafe { release_session(); }
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn release_session() {
    let mut session = SESSION.lock().unwrap();
    *session = None;
}

/// Detects a face from an image file and returns bounding box information.
///
/// This function loads an image from the filesystem, runs face detection, and writes the
/// resulting face bounding box to the provided output pointer. If multiple faces are detected,
/// it returns an error. Use this for images containing exactly one face.
///
/// # Arguments
/// * `image_path` - C string pointer to the image file path (supports PNG, JPEG, etc.)
/// * `out_box` - Pointer to an uninitialized `FaceBox` structure where results will be written
///
/// # Returns
/// * `0` - Success, face detected and `out_box` contains valid bounding box data
/// * Non-zero error code from `FaceDetectionResponse`:
///   - `InvalidImagePath` - Image file does not exist or path is invalid
///   - `ImageLoadingError` - Failed to load or decode the image
///   - `SessionGuardError` - Session not initialized (call [`init_session`] first)
///   - `NoFaceDetected` - No face found in the image
///   - `MultipleFaceDetected` - Multiple faces detected (function requires single face)
///   - `ImagePreprocessingError` - Failed to preprocess image for inference
///   - `InferenceError` - ONNX model inference failed
///   - `PostProcessingError` - Failed to parse model output
///
/// # Safety
/// This function is `unsafe extern "C"` and requires:
/// - `image_path` must be a valid null-terminated UTF-8 C string
/// - `out_box` must be a valid, non-null pointer to writable memory of size `sizeof(FaceBox)`
/// - The memory at `out_box` must not be accessed concurrently by other threads
/// - The caller must ensure proper alignment for `FaceBox` structure
///
/// # Panics
/// May panic if the global session mutex is poisoned or on UTF-8 conversion failure.
///
/// # Examples
/// ```rust,no_run
/// use std::ffi::CString;
/// use facedetector::models::FaceBox;
/// use facedetector::ffi::{init_session, detect_face_from_file};
///
/// unsafe {
///     init_session(CString::new("model.onnx").unwrap().as_ptr());
///     
///     let mut box_result = FaceBox::default();
///     let path = CString::new("face.jpg").unwrap();
///     let result = detect_face_from_file(path.as_ptr(), &mut box_result);
///     
///     if result == 0 {
///         println!("Face detected at ({}, {})", box_result.x1, box_result.y1);
///     }
///     
///     release_session();
/// }
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn detect_face_from_file(image_path: *const c_char, out_box: *mut FaceBox) -> c_int {
    let image_path = match convert_cstring_to_path(image_path) {
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

/// Detects a face from an in-memory image buffer and returns bounding box information.
///
/// This function processes an image from a byte buffer (e.g., JPEG/PNG encoded data or raw pixels),
/// runs face detection, and writes the resulting face bounding box to the provided output pointer.
/// Useful when images are already loaded in memory or received over network.
///
/// # Arguments
/// * `bytes` - Pointer to the image data buffer
/// * `len` - Length of the buffer in bytes
/// * `out_box` - Pointer to an uninitialized `FaceBox` structure where results will be written
///
/// # Returns
/// * `0` - Success, face detected and `out_box` contains valid bounding box data
/// * Non-zero error code from `FaceDetectionResponse`:
///   - `EmptyInputImage` - `bytes` is null or `len` is zero
///   - `ImageLoadingError` - Failed to decode image from buffer (unsupported format or corrupt data)
///   - `SessionGuardError` - Session not initialized (call [`init_session`] first)
///   - `NoFaceDetected` - No face found in the image
///   - `MultipleFaceDetected` - Multiple faces detected (function requires single face)
///   - `ImagePreprocessingError` - Failed to preprocess image for inference
///   - `InferenceError` - ONNX model inference failed
///   - `PostProcessingError` - Failed to parse model output
///
/// # Safety
/// This function is `unsafe extern "C"` and requires:
/// - `bytes` must be a valid pointer to at least `len` readable bytes, or null only when `len` is 0
/// - `out_box` must be a valid, non-null pointer to writable memory of size `sizeof(FaceBox)`
/// - The memory at `bytes` must remain valid and immutable for the duration of the call
/// - The caller must ensure proper alignment for both pointers
/// - No other thread may write to `bytes` during the call
/// - The memory at `out_box` must not be accessed concurrently by other threads
///
/// # Supported Image Formats
/// The function supports all formats supported by the `image` crate, including:
/// PNG, JPEG, GIF, BMP, TIFF, and WebP.
///
/// # Examples
/// ```rust,no_run
/// use facedetector::models::FaceBox;
/// use facedetector::ffi::{init_session, detect_face_from_memory};
/// use std::fs;
///
/// unsafe {
///     init_session(CString::new("model.onnx").unwrap().as_ptr());
///     
///     let image_data = fs::read("face.jpg").unwrap();
///     let mut box_result = FaceBox::default();
///     let result = detect_face_from_memory(
///         image_data.as_ptr(),
///         image_data.len(),
///         &mut box_result
///     );
///     
///     if result == 0 {
///         println!("Face confidence: {}", box_result.confidence);
///     }
///     
///     release_session();
/// }
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn detect_face_from_memory(
    bytes: *const u8,
    len: usize,
    out_box: *mut FaceBox,
) -> c_int {
    if len == 0 {
        return FaceDetectionResponse::ZeroLengthBytesError as c_int;
    }
    if bytes.is_null() {
        return FaceDetectionResponse::EmptyInputImageError as c_int;
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
