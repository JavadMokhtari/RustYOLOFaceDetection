# YOLO Face Detector 

A cross-language face detection library written in Rust, powered by ONNX Runtime and a YOLOv11 model. Exposes a C FFI
interface so it can be called from C, C++, Python, and any other language with C interop support.

---

## Features

- Single-face detection with bounding box and confidence score
- ONNX Runtime backend with optional CUDA GPU acceleration
- C-compatible API for cross-language use
- Prebuilt Windows SDK with DLL, header, and example scripts
- YOLOv11s-face model included

---

## Repository Structure

```
├── src/                    # Rust library source
│   ├── lib.rs              # C FFI entry points
│   ├── detector.rs         # ONNX inference pipeline
│   ├── models.rs           # FaceBox and response types
│   ├── configs.rs          # Model constants (size, thresholds)
│   └── utils.rs            # IoU, path helpers
│
├── sdk/                    # Prebuilt SDK for consumers
│   ├── bin/windows/        # Prebuilt DLL for Windows
│   ├── models/             # ONNX model file
│   ├── assets/             # Sample images for testing
│   ├── scripts/
│   │   ├── c/              # C example + Makefile + header
│   │   └── python/         # Python example
│   └── docs/               # Report / documentation
│
└── build.rs                # Cargo build script
```

---

## Usage Examples

- **Windows prebuilt** — DLL and import lib should place in [`sdk/bin/windows/`](sdk/bin/windows/). No Rust toolchain
  needed.

- **Model** — The included model is **YOLOv11s-face**, a lightweight single-class detector fine-tuned for faces with
  ONNX format.

### C

Build with the provided Makefile:

```bash
cd sdk/scripts/c
nmake
```

### Python

```bash
cd sdk/scripts/python
python main.py
```

---

## License

MIT
