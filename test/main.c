// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>
// #include "facedetector.h"

// // Helper: load entire file into a heap buffer
// unsigned char *load_file(const char *filename, size_t *out_len)
// {
//     FILE *file = fopen(filename, "rb");
//     if (!file)
//     {
//         perror("fopen failed");
//         return NULL;
//     }

//     fseek(file, 0, SEEK_END);
//     long file_size = ftell(file);
//     fseek(file, 0, SEEK_SET);
//     if (file_size <= 0)
//     {
//         fclose(file);
//         fprintf(stderr, "Empty or invalid file\n");
//         return NULL;
//     }

//     unsigned char *buffer = (unsigned char *)malloc(file_size);
//     if (!buffer)
//     {
//         fclose(file);
//         fprintf(stderr, "malloc failed\n");
//         return NULL;
//     }

//     size_t read_bytes = fread(buffer, 1, file_size, file);
//     fclose(file);

//     if (read_bytes != (size_t)file_size)
//     {
//         free(buffer);
//         fprintf(stderr, "Read size mismatch\n");
//         return NULL;
//     }

//     *out_len = read_bytes;
//     return buffer;
// }

// int main()
// {
//     // 1. Initialize the model session
//     int ret = init_session("yolov11s-face.onnx");
//     if (ret != 0)
//     {
//         printf("Failed to init session (code %d)\n", ret);
//         return -1;
//     }

//     // 2. Load an image file into memory
//     const char *image_path = "cool_girl.jpg";
//     size_t image_len = 0;
//     unsigned char *image_data = load_file(image_path, &image_len);
//     if (!image_data)
//     {
//         printf("Could not load %s\n", image_path);
//         release_session();
//         return -1;
//     }

//     // 3. Detect face from memory
//     FaceBox box;
//     int exec_code = detect_face_from_memory(image_data, image_len, &box);
//     printf("detect_face_from_memory returned %d\n", exec_code);

//     if (exec_code == 0)
//     {
//         printf("Detected face:\n");
//         printf("  x1 = %.2f, y1 = %.2f\n", box.x1, box.y1);
//         printf("  x2 = %.2f, y2 = %.2f\n", box.x2, box.y2);
//         printf("  confidence = %.4f\n", box.confidence);
//     }
//     else
//     {
//         printf("Detection failed with error code %d\n", exec_code);
//     }

//     // 4. Cleanup
//     free(image_data);
//     release_session();

//     return 0;
// }

// ##################################################################################

// #include <stdio.h>
// #include "facedetector.h"

// int main()
// {

//     int ret =
//         init_session("yolov11s-face.onnx");

//     if (ret != 0)
//     {
//         printf("Init failed\n");
//         return -1;
//     }

//     FaceBox box;
//     int exe_code = detect_face_from_file("two_face_boy.jpg", &box);
//     printf("Inference executed with code %d\n", exe_code);

//     if (exe_code == 0)
//     {
//         printf(
//             "Detected Bounding Box: %.2f %.2f %.2f %.2f conf=%.2f\n",
//             box.x1, box.y1, box.x2, box.y2, box.confidence);
//     }

//     release_session();

//     return 0;
// }

// ##################################################################################

#include <stdio.h>
#include "facedetector.h"

int main()
{
    int ret = init_session("yolov11s-face.onnx");

    if (ret != 0)
    {
        printf("Init failed: %d\n", ret);
        return -1;
    }

    FaceBox boxes[1];

    int exe_code = detect_face_from_file("cool_girl.jpg", &boxes[0]);
    printf("Process executed: %d\n", exe_code);

    if (exe_code == 0)
    {
        for (int i = 0; i < 1; i++)
        {

            printf(
                "Box %d: %.2f %.2f %.2f %.2f conf=%.2f\n",
                i + 1,
                boxes[i].x1,
                boxes[i].y1,
                boxes[i].x2,
                boxes[i].y2,
                boxes[i].confidence);
        }
    };

    release_session();
    return 0;
}