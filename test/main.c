#include <stdio.h>
#include "facedetector.h"

int main()
{

    int ret =
        init_session("yolov11s-face.onnx");

    if (ret != 0)
    {
        printf("Init failed\n");
        return -1;
    }

    FaceBox boxes[100];

    int count = detect_faces("cool_girl.jpg", boxes);

    printf("Detected: %d\n", count);

    for (int i = 0; i < count; i++)
    {

        printf(
            "Box %d: %.2f %.2f %.2f %.2f conf=%.2f\n",
            i,
            boxes[i].x1,
            boxes[i].y1,
            boxes[i].x2,
            boxes[i].y2,
            boxes[i].confidence);
    }

    release_session();

    return 0;
}