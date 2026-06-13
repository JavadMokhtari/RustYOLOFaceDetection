use crate::models::{FaceBox, FaceDetectionResponse};
use std::ffi::{CStr, c_int};
use std::os::raw::c_char;
use std::path::Path;

pub fn iou(a: &FaceBox, b: &FaceBox) -> f32 {
    let inter = intersection(a, b);
    let union = union(a, b);
    if union == 0.0 { 0.0 } else { inter / union }
}

fn intersection(a: &FaceBox, b: &FaceBox) -> f32 {
    let x1 = a.x1.max(b.x1);
    let y1 = a.y1.max(b.y1);
    let x2 = a.x2.min(b.x2);
    let y2 = a.y2.min(b.y2);
    if x1 >= x2 || y1 >= y2 {
        0.0
    } else {
        (x2 - x1) * (y2 - y1)
    }
}

fn union(a: &FaceBox, b: &FaceBox) -> f32 {
    let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    area_a + area_b - intersection(a, b)
}

pub fn convert_cstring_to_path<'a>(path_ptr: *const c_char) -> Result<&'a Path, c_int> {
    if path_ptr.is_null() {
        return Err(FaceDetectionResponse::NullPointerError as c_int);
    }

    let path_str = unsafe { CStr::from_ptr(path_ptr) }
        .to_str()
        .map_err(|_| FaceDetectionResponse::PathUTF8Error as c_int)?;

    let path = Path::new(path_str);
    Ok(path)
}
