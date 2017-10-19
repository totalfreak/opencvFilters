#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

int main() {

    Mat img, edges;
    img = imread("/home/daniel/CLionProjects/opencvFilters/Lenna.png", CV_LOAD_IMAGE_UNCHANGED);

    if ( !img.data ) {
        printf("No image data \n");
        return -1;
    }

    cvtColor(img, edges, CV_BGR2GRAY);

    GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
    Canny(edges, edges, 0, 30, 3);

    imshow("Edges", edges);

    namedWindow("Lenna", WINDOW_AUTOSIZE);
    imshow("Lenna", img);

    waitKey(0);

    return 0;
}