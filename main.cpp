#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

int main() {

    VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    Mat img, edges;
    //img = imread("/home/daniel/Documents/opencvFilters/Lenna.png", CV_LOAD_IMAGE_UNCHANGED);

    for(;;) {
        Mat frame;
        cap >> frame;
        cvtColor(frame, edges, CV_BGR2GRAY);

        GaussianBlur(edges, edges, Size(5, 5), 1.5, 1.5);
        Canny(edges, edges, 0, 60, 3);

        imshow("Edges", edges);

        //namedWindow("Lenna", WINDOW_AUTOSIZE);
        //imshow("Lenna", img);

        if(waitKey(30) >= 0)
            break;
    }

    return 0;
}