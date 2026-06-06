use crate::models::FaceBox;
use std::ffi::CStr;
use std::os::raw::c_char;

pub fn intersection(box1: &FaceBox, box2: &FaceBox) -> f32 {
    (box1.x2.min(box2.x2) - box1.x1.max(box2.x1)) * (box1.y2.min(box2.y2) - box1.y1.max(box2.y1))
}

pub fn union(box1: &FaceBox, box2: &FaceBox) -> f32 {
    ((box1.x2 - box1.x1) * (box1.y2 - box1.y1)) + ((box2.x2 - box2.x1) * (box2.y2 - box2.y1))
        - intersection(box1, box2)
}

pub fn get_path_from_cstr<'a>(path_ptr: *const c_char) -> Result<&'a str, i32> {
    if path_ptr.is_null() {
        return Err(-1);
    }

    unsafe { CStr::from_ptr(path_ptr) }
        .to_str()
        .map_err(|_| -2)
}