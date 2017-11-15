#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>

using namespace cv;

int main() {


    Mat img, imgGray;
    img = imread("/home/daniel/Documents/opencvFilters/stars.jpeg", CV_LOAD_IMAGE_UNCHANGED);

    if (img.empty()) return -1;

    //Just blurring a little to remove some of the stars and noise
    GaussianBlur(img, img, Size(5, 5), 1.5, 1.5);

    //The image is really high resolution for a making many windows, so I use the opencv function resize, to make it more manageable
    resize(img, img, Size(640, 480));

    //Converting image to grayscale
    cvtColor(img, imgGray, CV_BGR2GRAY);


    //Standard Sobel kernel
    int kernelX[3][3] = {1, 0, -1, 2, 0, -2, 1, 0, -1};

    //Sobel-Feldman kernel
    //int kernelX[3][3] = {3, 0, -3, -10, 0, -10, 3, 0, -3};

    //Standard Sobel kernel
    int kernelY[3][3] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

    //Sobel-Feldman kernel
    //int kernelY[3][3] = {3, 10, 3, 0, 0, 0, -3, -10, -3};

    int radius = 1;

    Mat src = imgGray.clone();

    //Saving the current "initial" image
    Mat gradX = imgGray.clone();
    Mat gradY = imgGray.clone();
    Mat gradF = imgGray.clone();

    //Looping over the the image with the x kernel
    for (int r = radius; r < src.rows - radius; ++r) {
        for (int c = radius; c < src.cols - radius; ++c) {
            int s = 0;
            for (int i = -radius; i <= radius; ++i) {
                for (int j = -radius; j <= radius; ++j) {
                    s += src.at<uchar>(r + i, c + j) * kernelX[i + radius][j + radius];
                }
            }
            gradX.at<uchar>(r - radius, c - radius) = static_cast<uchar>(s / 30);
        }
    }

    //Taking the absolute values from the image and scaling the whole new image with it
    Mat absGradX;
    convertScaleAbs(gradX, absGradX);
    imshow("X edge detection", absGradX);

    //Looping over the image with the y kernel
    for (int r = radius; r < src.rows - radius; ++r) {
        for (int c = radius; c < src.cols - radius; ++c) {
            int s = 0;

            for (int i = -radius; i <= radius; ++i) {
                for (int j = -radius; j <= radius; ++j) {
                    s += src.at<uchar>(r + i, c + j)* kernelY[i + radius][j + radius];
                }
            }
            gradY.at<uchar>(r - radius, c - radius) = static_cast<uchar>(s / 30);
        }
    }

    //Taking the absolute values from the image and scaling the whole new image with it
    Mat absGradY;
    convertScaleAbs(gradY, absGradY);
    imshow("Y edge detection", absGradY);

    //Loop that adds the two arrays together
    //This means that the "edges will have been found on both the x and the y"
    for (int i = 0; i < gradF.rows; i++) {
        for (int j = 0; j < gradF.cols; j++) {
            //Have to cast to uchar, because normals ints don't work
            gradF.at<uchar>(i, j) = static_cast<uchar>(sqrt(pow(gradX.at<uchar>(i, j), 2) + pow(gradY.at<uchar>(i, j), 2)));

            if (gradF.at<uchar>(i, j) > 240) {
                gradF.at<uchar>(i, j) = 255;
            } else {
                gradF.at<uchar>(i, j) = 0;
            }
        }
    }

    imshow("Edges", gradF);

    /*

    //Canny(edges, edges, 0, 60, 3);
    Sobel(edges, edges, 1, -1, -1);

    imshow("Edges", edges);

    //namedWindow("Lenna", WINDOW_AUTOSIZE);
    //imshow("Lenna", img);
    */
    waitKey(0);


    return 0;
}