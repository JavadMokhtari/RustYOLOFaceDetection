#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FaceBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
}

#[repr(i32)]
#[derive(Debug, Copy, Clone)]
pub enum FaceDetectionResponse {
    Success = 0,
    // IO Errors
    NullPointerError = 10,
    ZeroLengthBytesError = 11,
    PathUTF8Error = 12,
    InvalidONNXModelPath = 13,
    InvalidImagePath = 14,
    EmptyInputImageError = 15,
    // Session Errors
    SessionGuardError = 20,
    SessionBuilderError = 21,
    SessionInitializationError = 22,
    ExecutionProviderError = 23,
    ONNXModelLoadingError = 24,
    // Processing Errors
    ImageLoadingError = 30,
    ImagePreprocessingError = 31,
    InferenceError = 32,
    PostProcessingError = 33,
    // Result Errors
    NoFaceDetected = 41,
    MultipleFaceDetected = 42,
}
