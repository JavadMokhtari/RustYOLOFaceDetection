import ctypes
from pathlib import Path


class FaceBox(ctypes.Structure):
    _fields_ = [
        ("x1", ctypes.c_float),
        ("y1", ctypes.c_float),
        ("x2", ctypes.c_float),
        ("y2", ctypes.c_float),
        ("confidence", ctypes.c_float),
    ]


cur_dir = Path(__file__).parent

lib = ctypes.CDLL(str(cur_dir / "facedetector.dll"))

lib.init_session.argtypes = [ctypes.c_char_p]
lib.init_session.restype = ctypes.c_int

lib.detect_faces.argtypes = [
    ctypes.c_char_p,
    ctypes.POINTER(FaceBox),
]

lib.detect_faces.restype = ctypes.c_int

model_path = str(cur_dir / "yolov11s-face.onnx").encode("utf-8")
image_path = str(cur_dir / "underexposed-vs-overexposed.jpg").encode("utf-8")

# Initialize ONNX session
ret = lib.init_session(model_path)
print("init:", ret)

boxes = (FaceBox * 100)()
count = lib.detect_faces(image_path, boxes)
print("count:", count)

for i in range(count):
    print(
        boxes[i].x1,
        boxes[i].y1,
        boxes[i].x2,
        boxes[i].y2,
        boxes[i].confidence,
    )

# Cleanup
lib.release_session()
