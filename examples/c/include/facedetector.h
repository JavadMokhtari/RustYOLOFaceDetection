#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H

#include <stddef.h> // for size_t

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief Result codes for face detection operations.
     *
     * These values exactly match the Rust `FaceDetectionResponse` enum
     * (`#[repr(i32)]`) for seamless FFI compatibility.
     */
    typedef enum FaceDetectionResponse
    {
        /* Success */
        SUCCESS = 0,

        /* IO / Input Errors (10-19) */
        NULL_POINTER = 10,
        PATH_UTF8_ERROR = 11,
        INVALID_ONNX_PATH = 12,
        INVALID_IMAGE_PATH = 13,
        EMPTY_INPUT_IMAGE = 14,

        /* Session Errors (20-29) */
        SESSION_GUARD_ERROR = 20,
        SESSION_BUILDER_ERROR = 21,
        SESSION_INIT_ERROR = 22,
        EXECUTION_PROVIDER_ERROR = 23,
        ONNX_LOADING_ERROR = 24,

        /* Processing Errors (30-39) */
        IMAGE_LOADING_ERROR = 30,
        PREPROCESSING_ERROR = 31,
        INFERENCE_ERROR = 32,
        POSTPROCESSING_ERROR = 33,

        /* Detection Result Errors (40-49) */
        NO_FACE = 41,
        MULTIPLE_FACES = 42,
    } FaceDetectionResponse;

    /**
     * @brief Face bounding box with confidence score.
     */
    typedef struct
    {
        float x1;         /**< Top-left X coordinate */
        float y1;         /**< Top-left Y coordinate */
        float x2;         /**< Bottom-right X coordinate */
        float y2;         /**< Bottom-right Y coordinate */
        float confidence; /**< Confidence score (0.0 - 1.0) */
    } FaceBox;

    /**
     * @brief Initialize the ONNX session with the given model.
     *
     * @param model_path Path to the .onnx model file.
     * @return FACE_DETECT_SUCCESS on success, otherwise error code.
     */
    FaceDetectionResponse init_session(const char *model_path);

    /**
     * @brief Detect face from an image file.
     *
     * @param image_path Path to the image file (supported formats: JPEG, PNG, etc.).
     * @param out_box Pointer to a FaceBox structure that will be filled on success.
     * @return FACE_DETECT_SUCCESS if exactly one face was detected and returned,
     *         FACE_DETECT_NO_FACE or FACE_DETECT_MULTIPLE_FACES otherwise,
     *         or other error codes.
     */
    FaceDetectionResponse detect_face_from_file(const char *image_path, FaceBox *out_box);

    /**
     * @brief Detect face from image data in memory.
     *
     * @param bytes Pointer to image bytes (JPEG, PNG, etc.).
     * @param len   Size of the image data in bytes.
     * @param out_box Pointer to a FaceBox structure that will be filled on success.
     * @return Same as detect_face_from_file().
     */
    FaceDetectionResponse detect_face_from_memory(const unsigned char *bytes, size_t len, FaceBox *out_box);

    /**
     * @brief the ONNX session and free associated resources.
     * Call this when the library is no longer needed (e.g. at program shutdown).
     */
    void release_session(void);

#ifdef __cplusplus
}
#endif

#endif /* FACEDETECTOR_H */