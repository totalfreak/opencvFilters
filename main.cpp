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
    imshow("Original gray scale image", src);
    //Saving the current "initial" image
    Mat gradX = imgGray.clone();
    Mat gradY = imgGray.clone();
    Mat gradF = imgGray.clone();

    //Looping over the the image with the x kernel
    //From this we get the gradient image
    for (int r = radius; r < src.rows - radius; ++r) {
        for (int c = radius; c < src.cols - radius; ++c) {
            int s = 0;
            for (int i = -radius; i <= radius; ++i) {
                for (int j = -radius; j <= radius; ++j) {
                    s += src.at<uchar>(r + i, c + j) * kernelX[i + radius][j + radius];
                }
            }
            //Honestly don't know why this works, but it does
            gradX.at<uchar>(r - radius, c - radius) = s / 30;
        }
    }

    //Converting the image to the absolute values
    //Takes the image back to 8 bit, very important.
    Mat absGradX;
    convertScaleAbs(gradX, absGradX);
    imshow("X edge detection", absGradX);
    //Looping over the image with the y kernel
    //From this we get the gradient image
    for (int r = radius; r < src.rows - radius; ++r) {
        for (int c = radius; c < src.cols - radius; ++c) {
            int s = 0;

            for (int i = -radius; i <= radius; ++i) {
                for (int j = -radius; j <= radius; ++j) {
                    s += src.at<uchar>(r + i, c + j)* kernelY[i + radius][j + radius];
                }
            }
            //Honestly don't know why this works, but it does
            gradY.at<uchar>(r - radius, c - radius) = s / 30;
        }
    }

    //Converting the image to the absolute values
    //Takes the image back to 8 bit, very important.
    Mat absGradY;
    convertScaleAbs(gradY, absGradY);
    imshow("Y edge detection", absGradY);


    //Here we calculate an approximation of the gradient at every point, using both the x and y images
    for (int i = 0; i < gradF.rows; i++) {
        for (int j = 0; j < gradF.cols; j++) {

            gradF.at<uchar>(i, j) = static_cast<uchar>(sqrt(pow(gradX.at<uchar>(i, j), 2) + pow(gradY.at<uchar>(i, j), 2)));
            //If the magnitude of the resulting pixel is higher than 240, max it
            //Else zero it, thus making the image binary, and kewl
            if (gradF.at<uchar>(i, j) > 240) {
                gradF.at<uchar>(i, j) = 255;
            } else {
                gradF.at<uchar>(i, j) = 0;
            }
        }
    }

    imshow("Edges", gradF);

    waitKey(0);


    return 0;
}