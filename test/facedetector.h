#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct
    {
        float x1;
        float y1;
        float x2;
        float y2;
        float confidence;
    } FaceBox;

    int init_session(const char *model_path);

    int detect_face_from_file(const char *image_path, FaceBox *out_boxes);
    int detect_face_from_memory(unsigned char *bytes, size_t len, FaceBox *out_boxes);

    void release_session();

#ifdef __cplusplus
}
#endif

#endif