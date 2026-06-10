import ctypes
from pathlib import Path


# ==========================================
# FaceBox Struct
# ==========================================

class FaceBox(ctypes.Structure):
    _fields_ = [
        ("x1", ctypes.c_float),
        ("y1", ctypes.c_float),
        ("x2", ctypes.c_float),
        ("y2", ctypes.c_float),
        ("confidence", ctypes.c_float),
    ]


# ==========================================
# Paths
# ==========================================

cur_dir = Path(__file__).parent

dll_path = cur_dir / "../../target/release/facedetector.dll"
model_path = cur_dir / "../../models/yolov11s-face.onnx"
image_path = cur_dir / "../../assets/cool_girl.jpg"


# ==========================================
# Load DLL
# ==========================================

lib = ctypes.CDLL(str(dll_path))


# ==========================================
# Function Signatures
# ==========================================

lib.init_session.argtypes = [
    ctypes.c_char_p
]
lib.init_session.restype = ctypes.c_int


lib.detect_face_from_file.argtypes = [
    ctypes.c_char_p,
    ctypes.POINTER(FaceBox),
]
lib.detect_face_from_file.restype = ctypes.c_int


lib.detect_face_from_memory.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.c_size_t,
    ctypes.POINTER(FaceBox),
]
lib.detect_face_from_memory.restype = ctypes.c_int


lib.release_session.argtypes = []
lib.release_session.restype = None


# ==========================================
# Initialize Session
# ==========================================

result = lib.init_session(
    str(model_path).encode("utf-8")
)

if result != 0:
    raise RuntimeError(
        f"init_session failed: {result}"
    )

print(
    "Session initialized successfully."
)


# ==========================================
# detect_face_from_file
# ==========================================

file_box = FaceBox()

result = lib.detect_face_from_file(
    str(image_path).encode("utf-8"),
    ctypes.byref(file_box),
)

if result == 0:

    print("\n[file] Face Detected:")

    print(f"x1: {file_box.x1:.2f}")
    print(f"y1: {file_box.y1:.2f}")
    print(f"x2: {file_box.x2:.2f}")
    print(f"y2: {file_box.y2:.2f}")

    print(
        f"confidence: "
        f"{file_box.confidence:.4f}"
    )

else:
    print(
        f"\ndetect_face_from_file "
        f"failed: {result}"
    )


# ==========================================
# detect_face_from_memory
# ==========================================

image_bytes = image_path.read_bytes()

buffer = (
    ctypes.c_ubyte * len(image_bytes)
).from_buffer_copy(image_bytes)

memory_box = FaceBox()

result = lib.detect_face_from_memory(
    buffer,
    len(image_bytes),
    ctypes.byref(memory_box),
)

if result == 0:

    print("\n[memory] Face Detected:")

    print(f"x1: {memory_box.x1:.2f}")
    print(f"y1: {memory_box.y1:.2f}")
    print(f"x2: {memory_box.x2:.2f}")
    print(f"y2: {memory_box.y2:.2f}")

    print(
        f"confidence: "
        f"{memory_box.confidence:.4f}"
    )

else:
    print(
        f"\ndetect_face_from_memory "
        f"failed: {result}"
    )


# ==========================================
# Release Session
# ==========================================

lib.release_session()

print("\nSession released.")