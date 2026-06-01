use image;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int};

#[unsafe(no_mangle)]
pub extern "C" fn detect_faces(image_path: *const c_char) -> c_int {
    // Safety check
    if image_path.is_null() {
        return -1;
    }

    // Convert C string → Rust string
    let c_str = unsafe { CStr::from_ptr(image_path) };

    let path = match c_str.to_str() {
        Ok(v) => v,
        Err(_) => return -2,
    };

    // Load image
    let img = match image::open(path) {
        Ok(v) => v,
        Err(_) => return -3,
    };

    // Dummy face detection logic
    // Replace later with ONNX inference

    let width = img.width();
    let height = img.height();

    if width > 100 {
        println!("Image width: {}", width);
        println!("Image height: {}", height);
        1
    } else {
        0
    }
}
