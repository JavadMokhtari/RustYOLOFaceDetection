use crate::configs::*;
use crate::errors::FaceDetectionResponse;
use crate::models::FaceBox;
use crate::utils::iou;
use image::DynamicImage;
use image::imageops::FilterType;
use ndarray::{Array4, Axis, s};
use ort::{session::Session, value::TensorRef};
use std::sync::{LazyLock, Mutex};

pub static SESSION: LazyLock<Mutex<Option<Session>>> = LazyLock::new(|| Mutex::new(None));

pub fn detect_face(img: DynamicImage) -> Result<FaceBox, i32> {
    let (orig_w, orig_h) = (img.width() as f32, img.height() as f32);

    // === Preprocessing ===
    let resized = img.resize_exact(IMG_SIZE as u32, IMG_SIZE as u32, FilterType::Triangle);
    let rgb = resized.into_rgb8(); // avoid DynamicImage overhead
    let raw = rgb.as_raw(); // &[u8]

    let inv255 = 1.0 / 255.0;
    let total_pixels = IMG_SIZE * IMG_SIZE;
    let mut input_vec = vec![0.0; total_pixels * 3];

    // Split the output vector into three channel slices
    let (r, rest) = input_vec.split_at_mut(total_pixels);
    let (g, b) = rest.split_at_mut(total_pixels);

    // Fill each channel by iterating over RGB triplets
    for (i, chunk) in raw.chunks_exact(3).enumerate() {
        r[i] = chunk[0] as f32 * inv255;
        g[i] = chunk[1] as f32 * inv255;
        b[i] = chunk[2] as f32 * inv255;
    }

    let input_arr = Array4::from_shape_vec((1, 3, IMG_SIZE, IMG_SIZE), input_vec)
        .map_err(|_| FaceDetectionResponse::ImagePreprocessingError as i32)?;

    let input_tensor = TensorRef::from_array_view(input_arr.view())
        .map_err(|_| FaceDetectionResponse::ImagePreprocessingError as i32)?;

    // === Inference ===
    let mut guard = SESSION.lock().unwrap();
    let session = guard
        .as_mut()
        .ok_or(FaceDetectionResponse::SessionGuardError as i32)?;

    let outputs = session
        .run(ort::inputs!["images" => input_tensor])
        .map_err(|_| FaceDetectionResponse::InferenceError as i32)?;

    // === Post-processing ===
    let output_tensor = outputs
        .get("output0")
        .ok_or(FaceDetectionResponse::PostProcessingError as i32)?
        .try_extract_array::<f32>()
        .map_err(|_| FaceDetectionResponse::PostProcessingError as i32)?;

    let output = output_tensor.t().into_owned(); // transpose once
    let detections = output.slice(s![.., .., 0]); // batch=0

    let scale_x = orig_w / IMG_SIZE as f32;
    let scale_y = orig_h / IMG_SIZE as f32;
    let confidence_threshold = CONFIDENCE;

    let mut detected_boxes = Vec::with_capacity(64);

    for row in detections.axis_iter(Axis(0)) {
        let prob = row[4_usize];
        if prob >= confidence_threshold {
            let xc = row[0_usize] * scale_x;
            let yc = row[1_usize] * scale_y;
            let w = row[2_usize] * scale_x;
            let h = row[3_usize] * scale_y;
            let half_w = w * 0.5;
            let half_h = h * 0.5;

            detected_boxes.push(FaceBox {
                x1: xc - half_w,
                y1: yc - half_h,
                x2: xc + half_w,
                y2: yc + half_h,
                confidence: prob,
            });
        }
    }

    // === NMS ===
    detected_boxes.sort_unstable_by(|a, b| b.confidence.total_cmp(&a.confidence));

    let mut faces = Vec::with_capacity(3);

    while let Some(best) = detected_boxes.first().copied() {
        faces.push(best);
        detected_boxes.retain(|b| iou(&best, b) < IOU);
    }

    match faces.len() {
        1 => Ok(faces[0]),
        0 => Err(FaceDetectionResponse::NoFaceDetected as i32),
        _ => Err(FaceDetectionResponse::MultipleFaceDetected as i32),
    }
}

// /// Batch face detection. Returns one result per input image.
// pub fn detect_face_from_images(images: Vec<DynamicImage>) -> Vec<Result<FaceBox, i32>> {
//     if images.is_empty() {
//         return vec![];
//     }
//
//     let batch_size = images.len();
//     let target_h = 640;
//     let target_w = 640;
//
//     // Preprocess all images into a single (N, 3, H, W) tensor
//     let mut input_vec = Vec::with_capacity(batch_size * 3 * target_h * target_w);
//
//     let mut original_sizes: Vec<(f32, f32)> = Vec::with_capacity(batch_size);
//
//     for img in &images {
//         let (orig_w, orig_h) = (img.width() as f32, img.height() as f32);
//         original_sizes.push((orig_w, orig_h));
//
//         let resized = img.resize_exact(target_w as u32, target_h as u32, FilterType::Triangle);
//         let rgb = resized.to_rgb8();
//         let raw_pixels = rgb.into_raw();
//
//         // CHW order, normalized [0,1]
//         for c in 0..3 {
//             for y in 0..target_h {
//                 for x in 0..target_w {
//                     let idx = (y * target_w + x) * 3 + c;
//                     let pixel = raw_pixels[idx] as f32 / 255.0;
//                     input_vec.push(pixel);
//                 }
//             }
//         }
//     }
//
//     let input_arr = Array4::from_shape_vec((batch_size, 3, target_h, target_w), input_vec)
//         .expect("Failed to create input array");
//
//     let input_tensor =
//         TensorRef::from_array_view(input_arr.view()).expect("Failed to create input tensor");
//
//     // Run batched inference
//     let mut session_guard = SESSION.lock().unwrap();
//     let session = session_guard.as_mut().expect("Session not initialized");
//
//     let start = Instant::now();
//     let outputs = session
//         .run(ort::inputs!["images" => input_tensor])
//         .expect("Failed to inference.");
//
//     // Parse outputs
//     let output = outputs
//         .get("output0")
//         .unwrap()
//         .try_extract_array::<f32>()
//         .unwrap()
//         .t()
//         .into_owned();
//
//     let elapsed = start.elapsed();
//     println!("ONNX inference time: {:?}", elapsed);
//
//     let mut results = Vec::with_capacity(batch_size);
//
//     for b in 0..batch_size {
//         let img_output = output.slice(s![.., .., b]);
//
//         let (img_width, img_height) = original_sizes[b];
//
//         let mut detected_boxes = Vec::new();
//
//         for row in img_output.axis_iter(Axis(0)) {
//             let row: Vec<f32> = row.iter().copied().collect();
//
//             // Find best class probability (skip the 4 box coords)
//             let (_, prob) = row
//                 .iter()
//                 .skip(4)
//                 .enumerate()
//                 .map(|(i, &v)| (i, v))
//                 .reduce(|a, b| if b.1 > a.1 { b } else { a })
//                 .unwrap_or((0, 0.0));
//
//             if prob < 0.5 {
//                 continue;
//             }
//
//             let xc = row[0] / target_w as f32 * img_width;
//             let yc = row[1] / target_h as f32 * img_height;
//             let w = row[2] / target_w as f32 * img_width;
//             let h = row[3] / target_h as f32 * img_height;
//
//             detected_boxes.push(FaceBox {
//                 x1: xc - w / 2.0,
//                 y1: yc - h / 2.0,
//                 x2: xc + w / 2.0,
//                 y2: yc + h / 2.0,
//                 confidence: prob,
//             });
//         }
//         // 4. NMS per image
//         detected_boxes.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
//
//         let mut result = Vec::new();
//         let iou_threshold = 0.45;
//
//         while let Some(best) = detected_boxes.first().copied() {
//             result.push(best);
//             detected_boxes.retain(|b| intersection(&best, b) / union(&best, b) < iou_threshold);
//         }
//
//         let res = match result.len() {
//             0 => Err(2), // No face
//             1 => Ok(result[0]),
//             _ => Err(3), // multiple faces
//         };
//         results.push(res);
//     }
//     results
// }
