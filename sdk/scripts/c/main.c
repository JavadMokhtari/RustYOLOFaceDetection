#include <stdio.h>
#include <stdlib.h>

#include "facedetector.h"

int main()
{
    printf("facedetector v%s\n", facedetector_version());

    const char *model_path = "../../models/yolov11s-face.onnx";
    const char *image_path = "../../assets/cool_girl.jpg";

    FaceDetectionResponse result;

    // Initialize ONNX session
    result = init_session(model_path);

    if (result != SUCCESS)
    {
        printf("init_session failed with code: %d\n", result);
        return -1;
    }

    printf("Session initialized successfully.\n\n");

    // Output face box
    FaceBox box;

    // Run face detection from file
    result = detect_face_from_file(image_path, &box);

    if (result != SUCCESS)
    {
        printf("Face detection from file failed with code: %d\n", result);
        release_session();
        return -1;
    }

    // Print detection result
    printf("\nFace detected from file:\n{\n");
    printf("\tx1: %.2f\n", box.x1);
    printf("\ty1: %.2f\n", box.y1);
    printf("\tx2: %.2f\n", box.x2);
    printf("\ty2: %.2f\n", box.y2);
    printf("\tConfidence: %.4f\n}\n", box.confidence);

    // Run face detection from memory
    FILE *fp = fopen(image_path, "rb");

    if (!fp)
    {
        printf("\nFailed to open image file.\n");
        release_session();
        return -1;
    }

    // Get file size
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    rewind(fp);

    // Allocate memory
    unsigned char *buffer = (unsigned char *)malloc(file_size);

    if (!buffer)
    {
        printf("\nMemory allocation failed.\n");
        fclose(fp);
        release_session();
        return -1;
    }

    // Read image bytes
    fread(buffer, 1, file_size, fp);
    fclose(fp);

    FaceBox mem_box;
    result = detect_face_from_memory(buffer, file_size, &mem_box);

    if (result == SUCCESS)
    {
        printf("\nFace detected from file:\n{\n");
        printf("\tx1: %.2f\n", mem_box.x1);
        printf("\ty1: %.2f\n", mem_box.y1);
        printf("\tx2: %.2f\n", mem_box.x2);
        printf("\ty2: %.2f\n", mem_box.y2);
        printf("\tConfidence: %.4f\n}\n", mem_box.confidence);
    }
    else
    {
        printf("\nFace detection from memory failed: %d\n", result);
    }

    // Free image buffer
    free(buffer);

    // Release resources
    release_session();
    printf("\nSession closed.");

    return 0;
}