use image::imageops::FilterType;
use ndarray::{s, Array4, Axis};
use once_cell::sync::Lazy;
use ort::{ep::CUDA, session::Session, value::TensorRef};
use std::{
    ffi::CStr,
    os::raw::{c_char, c_int},
    sync::Mutex,
};

static SESSION: Lazy<Mutex<Option<Session>>> = Lazy::new(|| Mutex::new(None));

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FaceBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
}

pub fn intersection(box1: &FaceBox, box2: &FaceBox) -> f32 {
    (box1.x2.min(box2.x2) - box1.x1.max(box2.x1)) * (box1.y2.min(box2.y2) - box1.y1.max(box2.y1))
}

pub fn union(box1: &FaceBox, box2: &FaceBox) -> f32 {
    ((box1.x2 - box1.x1) * (box1.y2 - box1.y1)) + ((box2.x2 - box2.x1) * (box2.y2 - box2.y1))
        - intersection(box1, box2)
}

#[unsafe(no_mangle)]
pub extern "C" fn init_session(model_path: *const c_char) -> c_int {
    if model_path.is_null() {
        return -1;
    }

    let c_str = unsafe { CStr::from_ptr(model_path) };

    let model_path = match c_str.to_str() {
        Ok(v) => v,
        Err(_) => return -2,
    };

    let session = match Session::builder() {
        Ok(builder) => {
            let mut builder = match builder.with_execution_providers([CUDA::default().build()]) {
                Ok(v) => v,
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

#[unsafe(no_mangle)]
pub extern "C" fn detect_faces(image_path: *const c_char, out_boxes: *mut FaceBox) -> c_int {
    if image_path.is_null() {
        return -1;
    }

    let c_str = unsafe { CStr::from_ptr(image_path) };

    let image_path = match c_str.to_str() {
        Ok(v) => v,
        Err(_) => return -2,
    };

    let img = match image::open(image_path) {
        Ok(v) => v,
        Err(_) => return -3,
    };

    // Image pre-processing
    let resized = img.resize_exact(640, 640, FilterType::Triangle);
    let rgb = resized.to_rgb8();
    let mut input = Array4::<f32>::zeros((1, 3, 640, 640));

    for y in 0..640 {
        for x in 0..640 {
            let pixel = rgb.get_pixel(x, y);

            input[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;

            input[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;

            input[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
        }
    }

    let mut session_guard = SESSION.lock().unwrap();

    let session = match session_guard.as_mut() {
        Some(v) => v,
        None => return -10,
    };

    let input_tensor = match TensorRef::from_array_view(input.view()) {
        Ok(v) => v,
        Err(_) => return -11,
    };

    let outputs = match session.run(ort::inputs!["images" => input_tensor]) {
        Ok(v) => v,
        Err(_) => return -12,
    };

    println!("Inference success");

    let output = outputs
        .get("output0")
        .unwrap()
        .try_extract_array::<f32>()
        .unwrap()
        .t()
        .into_owned();

    let output = output.slice(s![.., .., 0]);

    let (img_width, img_height) = (img.width(), img.height());

    let mut detected_boxes = Vec::new();

    for row in output.axis_iter(Axis(0)) {
        let row: Vec<_> = row.iter().copied().collect();

        let (_, prob) = row
            .iter()
            .skip(4)
            .enumerate()
            .map(|(index, value)| (index, *value))
            .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
            .unwrap();

        if prob < 0.5 {
            continue;
        }

        let xc = row[0] / 640.0 * img_width as f32;
        let yc = row[1] / 640.0 * img_height as f32;
        let w = row[2] / 640.0 * img_width as f32;
        let h = row[3] / 640.0 * img_height as f32;

        detected_boxes.push(FaceBox {
            x1: xc - w / 2.0,
            y1: yc - h / 2.0,
            x2: xc + w / 2.0,
            y2: yc + h / 2.0,
            confidence: prob,
        });
    }

    // new lines
    detected_boxes.sort_by(|box1, box2| box2.confidence.total_cmp(&box1.confidence));
    let mut result = Vec::new();

    let iou = 0.45;
    while !detected_boxes.is_empty() {
        result.push(detected_boxes[0]);
        detected_boxes = detected_boxes
            .iter()
            .filter(|box1| {
                intersection(&detected_boxes[0], &box1) / union(&detected_boxes[0], &box1) < iou
            })
            .copied()
            .collect();
    }
    let count = result.len();

    // let postproc_time = now.elapsed() - preproc_time - inference_time;
    // println!("Postprocessing time: {:?}", postproc_time);

    unsafe {
        for i in 0..count {
            *out_boxes.add(i) = result[i];
        }
    }
    count as c_int
}

#[unsafe(no_mangle)]
pub extern "C" fn release_session() {
    let mut session = SESSION.lock().unwrap();

    *session = None;
}
