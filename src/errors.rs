#[repr(i32)]
#[derive(Debug, Copy, Clone)]
pub enum FaceDetectionResponse {
    Success = 0,
    // IO Errors
    NullPointerError = 10,
    PathUTF8Error = 11,
    InvalidONNXModelPath = 12,
    InvalidImagePath = 13,
    EmptyInputImage = 14,
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