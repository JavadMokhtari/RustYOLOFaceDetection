# Face Detector SDK

A lightweight face detection SDK implemented in Rust and exposed through a C-compatible API. The SDK uses a YOLOv11-based ONNX model and can be integrated into applications written in C, C++, Java (JNI/JNA), Python, C#, Go, and other languages capable of calling native libraries. (version: `0.1.0`)

---

## Features

* Single-face detection from image files
* Single-face detection from image bytes in memory
* ONNX Runtime backend
* C-compatible API
* Windows x64 support
* Fast inference with low memory footprint
* Language-agnostic integration through FFI

---

## Package Contents

```text
face-detector-sdk/
│
├── assets/
│   ├── boy_two_faces.jpg
│   └── cool_girl.jpg
│
├── bin/
│   └── windows/
│       ├── facedetector.dll
│       └── facedetector.dll.lib
│
├── models/
│   └── yolov11s-face.onnx
│
├── scripts/
│   ├── c
│   |   ├── include/
│   |   |   └── facedetector.h
│   |   ├── main.c
│   |   └── Makefile
│   └── python
│       └── facedetector.dll.lib
│
└── README.md
```
---

## Requirements

### Runtime

* Windows 10 or newer (64-bit)

### Development

* Microsoft Visual Studio 2022
* MSVC Compiler Toolchain
* x64 Native Tools Command Prompt for VS 2022

---

## Building the C Example

Open:

```text
x64 Native Tools Command Prompt for VS 2022
```

Navigate to:

```bat
cd face-detector-sdk-VERSION/scripts/c
```

Display help:
```bat
nmake
```

Build:

```bat
nmake build
```

Run:

```bat
nmake run
```

Clean generated files:

```bat
nmake clean
```

---


## API Reference

### FaceDetectionResponse init_session(const char* model_path)

Initializes the ONNX Runtime session and loads the specified face detection model.

#### Parameters

| Name       | Description                  |
| ---------- | ---------------------------- |
| model_path | Path to a YOLOv11 ONNX model |

#### Returns

| Code               | Description                      |
| ------------------ | -------------------------------- |
| SUCCESS            | Session initialized successfully |
| INVALID_ONNX_PATH  | Invalid model path               |
| ONNX_LOADING_ERROR | Failed to load model             |
| SESSION_INIT_ERROR | Session initialization failed    |

---

### FaceDetectionResponse detect_face_from_file(const char* image_path, FaceBox* out_box)

Detects a face from an image file.

#### Parameters

| Name       | Description              |
| ---------- | ------------------------ |
| image_path | Path to image file       |
| out_box    | Output face bounding box |

#### Returns

| Code                | Description                 |
| ------------------- | --------------------------- |
| SUCCESS             | Exactly one face detected   |
| NO_FACE             | No face found               |
| MULTIPLE_FACES      | More than one face detected |
| IMAGE_LOADING_ERROR | Failed to load image        |

---

### FaceDetectionResponse detect_face_from_memory(const unsigned char* bytes, size_t len, FaceBox* out_box)

Detects a face from image bytes stored in memory.

#### Parameters

| Name    | Description              |
| ------- | ------------------------ |
| bytes   | Pointer to image bytes   |
| len     | Image size in bytes      |
| out_box | Output face bounding box |

#### Returns

Same as `detect_face_from_file()`.

---

### void release_session(void)

Releases the ONNX Runtime session and all associated resources.

Call this function once when the SDK is no longer needed.

---

## FaceBox Structure

```c
typedef struct
{
    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
} FaceBox;
```

### Coordinates

```text
(x1, y1) ---------
   |             |
   |   FACE      |
   |             |
   --------- (x2, y2)
```

---

## Error Codes

### Success

| Value | Name    |
| ----- | ------- |
| 0     | SUCCESS |

### Input Errors

| Value | Name               |
| ----- | ------------------ |
| 10    | NULL_POINTER       |
| 11    | PATH_UTF8_ERROR    |
| 12    | INVALID_ONNX_PATH  |
| 13    | INVALID_IMAGE_PATH |
| 14    | EMPTY_INPUT_IMAGE  |

### Session Errors

| Value | Name                     |
| ----- | ------------------------ |
| 20    | SESSION_GUARD_ERROR      |
| 21    | SESSION_BUILDER_ERROR    |
| 22    | SESSION_INIT_ERROR       |
| 23    | EXECUTION_PROVIDER_ERROR |
| 24    | ONNX_LOADING_ERROR       |

### Processing Errors

| Value | Name                 |
| ----- | -------------------- |
| 30    | IMAGE_LOADING_ERROR  |
| 31    | PREPROCESSING_ERROR  |
| 32    | INFERENCE_ERROR      |
| 33    | POSTPROCESSING_ERROR |

### Detection Errors

| Value | Name           |
| ----- | -------------- |
| 41    | NO_FACE        |
| 42    | MULTIPLE_FACES |

---

## Thread Safety

The SDK currently maintains a global ONNX Runtime session.

Applications should initialize the session once and share it across threads.

Do not call `release_session()` while detection requests are still being processed.

---

## Model

The SDK ships with:

```text
models/yolov11s-face.onnx
```

The model file must be accessible when calling:

```c
init_session(model_path);
```

---

## Supported Image Formats

The SDK supports any image format supported by the underlying image decoding library, including:

* JPEG
* PNG
* BMP
* TIFF
* WebP

---

## Integration With Other Languages

The SDK can be integrated with:

* C
* C++
* Java (JNI|JNA)
* Python (ctypes)
* C#
* Go
* Rust

using the exported functions in `facedetector.dll`.

---

## License

MIT

