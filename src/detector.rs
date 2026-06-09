use crate::models::FaceBox;
use crate::utils::{intersection, union};
use image::DynamicImage;
use image::imageops::FilterType;
use ndarray::{Array4, Axis, s};
use once_cell::sync::Lazy;
use ort::{session::Session, value::TensorRef};
use std::sync::Mutex;
use std::time::Instant;

pub static SESSION: Lazy<Mutex<Option<Session>>> = Lazy::new(|| Mutex::new(None));


pub fn detect_face_from_image(img: DynamicImage) -> Result<FaceBox, i32> {
    let (orig_w, orig_h) = (img.width() as f32, img.height() as f32);

    // === Faster Preprocessing ===
    const IMG_SIZE: usize = 320;
    let resized = img.resize_exact(IMG_SIZE as u32, IMG_SIZE as u32, FilterType::Triangle);
    let rgb = resized.into_rgb8(); // avoid DynamicImage overhead

    let raw = rgb.as_raw(); // &[u8]

    // Build CHW f32 tensor with minimal allocations
    let mut input_vec = vec![0.0f32; 3 * IMG_SIZE * IMG_SIZE];
    let mut idx = 0;

    for c in 0..3 {
        for i in (c..raw.len()).step_by(3) {
            input_vec[idx] = raw[i] as f32 * (1.0 / 255.0);
            idx += 1;
        }
    }

    let input_arr = Array4::from_shape_vec((1, 3, IMG_SIZE, IMG_SIZE), input_vec).map_err(|_| -11)?;

    let input_tensor = TensorRef::from_array_view(input_arr.view()).map_err(|_| -11)?;

    // === Inference ===
    let mut guard = SESSION.lock().unwrap();
    let session = guard.as_mut().ok_or(-10)?;

    let outputs = session
        .run(ort::inputs!["images" => input_tensor])
        .map_err(|_| -12)?;

    // === Faster Post-processing ===
    let output_tensor = outputs
        .get("output0")
        .ok_or(-12)?
        .try_extract_array::<f32>()
        .map_err(|_| -12)?;

    let output = output_tensor.t().into_owned(); // transpose once
    let detections = output.slice(s![.., .., 0]); // batch=0

    let mut detected_boxes = Vec::new();

        for row in detections.axis_iter(Axis(0)) {
            let row: Vec<_> = row.iter().copied().collect();

            let (_, prob) = row
                .iter()
                .skip(4)
                .enumerate()
                .map(|(index, value)| (index, *value))
                .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
                .unwrap();

            if 0.85 < prob {

                let xc = row[0] / IMG_SIZE as f32 * orig_w;
                let yc = row[1] / IMG_SIZE as f32 * orig_h;
                let w = row[2] / IMG_SIZE as f32 * orig_w;
                let h = row[3] / IMG_SIZE as f32 * orig_h;

                detected_boxes.push(FaceBox {
                    x1: xc - w / 2.0,
                    y1: yc - h / 2.0,
                    x2: xc + w / 2.0,
                    y2: yc + h / 2.0,
                    confidence: prob,
                });
            }
        }

    // === NMS ===
    if detected_boxes.is_empty() {
        return Err(2);
    }

    detected_boxes.sort_unstable_by(|a, b| b.confidence.total_cmp(&a.confidence));

    let mut result = Vec::new();
    let iou_threshold = 0.45;

    while let Some(best) = detected_boxes.first().copied() {
        result.push(best);
        detected_boxes.retain(|b| intersection(&best, b) / union(&best, b) < iou_threshold);
    }

    match result.len() {
        1 => Ok(result[0]),
        0 => Err(2),
        _ => Err(3),
    }
}


// pub fn detect_face_from_image(img: DynamicImage) -> Result<FaceBox, i32> {
//     // Preprocess: resize and convert to raw RGB bytes
//     let resized = img.resize_exact(640, 640, FilterType::Triangle);
//     let rgb = resized.to_rgb8();
//     let raw_pixels = rgb.into_raw(); // Vec<u8> in RGBRGB... order
//
//     // Build CHW float array directly from raw pixels
//     let input_vec: Vec<f32> = (0..3)
//         .flat_map(|c| raw_pixels.iter().skip(c).step_by(3).map(|&p| p as f32 / 255.0))
//         .collect();
//
//     let input_arr = Array4::from_shape_vec((1, 3, 640, 640), input_vec).map_err(|_| -11)?;
//     let input_tensor = TensorRef::from_array_view(input_arr.view()).map_err(|_| -11)?;
//
//     // Run inference
//     let mut session_guard = SESSION.lock().unwrap();
//     let session = session_guard.as_mut().ok_or(-10)?;
//     let outputs = session
//         .run(ort::inputs!["images" => input_tensor])
//         .map_err(|_| -12)?;
//
//     // Parse outputs
//     let output = outputs
//         .get("output0")
//         .ok_or(-12)?
//         .try_extract_array::<f32>()
//         .map_err(|_| -12)?
//         .t()
//         .into_owned();
//     let output = output.slice(s![.., .., 0]);
//
//     let (img_width, img_height) = (img.width() as f32, img.height() as f32);
//     let mut detected_boxes = Vec::new();
//
//     for row in output.axis_iter(Axis(0)) {
//         let row: Vec<_> = row.iter().copied().collect();
//
//         let (_, prob) = row
//             .iter()
//             .skip(4)
//             .enumerate()
//             .map(|(index, value)| (index, *value))
//             .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
//             .unwrap();
//
//         if prob < 0.5 {
//             continue;
//         }
//
//         let xc = row[0] / 640.0 * img_width;
//         let yc = row[1] / 640.0 * img_height;
//         let w = row[2] / 640.0 * img_width;
//         let h = row[3] / 640.0 * img_height;
//
//         detected_boxes.push(FaceBox {
//             x1: xc - w / 2.0,
//             y1: yc - h / 2.0,
//             x2: xc + w / 2.0,
//             y2: yc + h / 2.0,
//             confidence: prob,
//         });
//     }
//
//     // Non‑maximum suppression
//     detected_boxes.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
//     let mut result = Vec::new();
//     let iou_threshold = 0.45;
//
//     while let Some(best) = detected_boxes.first().copied() {
//         result.push(best);
//         detected_boxes.retain(|b| intersection(&best, b) / union(&best, b) < iou_threshold);
//     }
//     match result.len() {
//         0 => Err(2),        // No face
//         1 => Ok(result[0]), // single face
//         _ => Err(3),        // multiple faces
//     }
// }

/// Batch face detection. Returns one result per input image.
pub fn detect_face_from_images(images: Vec<DynamicImage>) -> Vec<Result<FaceBox, i32>> {
    if images.is_empty() {
        return vec![];
    }

    let batch_size = images.len();
    let target_h = 640;
    let target_w = 640;

    // Preprocess all images into a single (N, 3, H, W) tensor
    let mut input_vec = Vec::with_capacity(batch_size * 3 * target_h * target_w);

    let mut original_sizes: Vec<(f32, f32)> = Vec::with_capacity(batch_size);

    for img in &images {
        let (orig_w, orig_h) = (img.width() as f32, img.height() as f32);
        original_sizes.push((orig_w, orig_h));

        let resized = img.resize_exact(target_w as u32, target_h as u32, FilterType::Triangle);
        let rgb = resized.to_rgb8();
        let raw_pixels = rgb.into_raw();

        // CHW order, normalized [0,1]
        for c in 0..3 {
            for y in 0..target_h {
                for x in 0..target_w {
                    let idx = (y * target_w + x) * 3 + c;
                    let pixel = raw_pixels[idx] as f32 / 255.0;
                    input_vec.push(pixel);
                }
            }
        }
    }

    let input_arr = Array4::from_shape_vec((batch_size, 3, target_h, target_w), input_vec)
        .expect("Failed to create input array");

    let input_tensor =
        TensorRef::from_array_view(input_arr.view()).expect("Failed to create input tensor");

    // Run batched inference
    let mut session_guard = SESSION.lock().unwrap();
    let session = session_guard.as_mut().expect("Session not initialized");

    let start = Instant::now();
    let outputs = session
        .run(ort::inputs!["images" => input_tensor])
        .expect("Failed to inference.");

    // Parse outputs
    let output = outputs
        .get("output0")
        .unwrap()
        .try_extract_array::<f32>()
        .unwrap()
        .t()
        .into_owned();

    let elapsed = start.elapsed();
    println!("ONNX inference time: {:?}", elapsed);

    let mut results = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        let img_output = output.slice(s![.., .., b]);

        let (img_width, img_height) = original_sizes[b];

        let mut detected_boxes = Vec::new();

        for row in img_output.axis_iter(Axis(0)) {
            let row: Vec<f32> = row.iter().copied().collect();

            // Find best class probability (skip the 4 box coords)
            let (_, prob) = row
                .iter()
                .skip(4)
                .enumerate()
                .map(|(i, &v)| (i, v))
                .reduce(|a, b| if b.1 > a.1 { b } else { a })
                .unwrap_or((0, 0.0));

            if prob < 0.5 {
                continue;
            }

            let xc = row[0] / target_w as f32 * img_width;
            let yc = row[1] / target_h as f32 * img_height;
            let w = row[2] / target_w as f32 * img_width;
            let h = row[3] / target_h as f32 * img_height;

            detected_boxes.push(FaceBox {
                x1: xc - w / 2.0,
                y1: yc - h / 2.0,
                x2: xc + w / 2.0,
                y2: yc + h / 2.0,
                confidence: prob,
            });
        }
        // 4. NMS per image
        detected_boxes.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
        // println!("detected boxes: {:?}", detected_boxes);

        let mut result = Vec::new();
        let iou_threshold = 0.45;

        while let Some(best) = detected_boxes.first().copied() {
            result.push(best);
            detected_boxes.retain(|b| intersection(&best, b) / union(&best, b) < iou_threshold);
        }

        let res = match result.len() {
            0 => Err(2), // No face
            1 => Ok(result[0]),
            _ => Err(3), // multiple faces
        };
        results.push(res);
    }
    results
}
