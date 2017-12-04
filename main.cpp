#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace cv;

Mat src, gradF, gradX, gradY;

void doImageProcessing() {

    imshow("original image", src);


    //Standard Sobel kernel
    //int kernelX[3][3] = {1, 0, -1, 2, 0, -2, 1, 0, -1};

    //Sobel-Feldman kernel
    int kernelX[3][3] = {-3, 0, 3, -10, 0, 10, -3, 0, 3};

    //Standard Sobel kernel
    //int kernelY[3][3] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

    //Sobel-Feldman kernel
    int kernelY[3][3] = {-3, -10, -3, 0, 0, 0, 3, 10, 3};


    int radius = 1;

    //Saving the initial image, to be overwritten by the for loops
    gradX = src.clone();
    gradY = src.clone();
    gradF = src.clone();

    //Looping over the the image with the x kernel
    //From this we get the gradient image
    for (int row = radius; row < src.rows - radius; row++) {
        for (int col = radius; col < src.cols - radius; col++) {
            int scale = 0;
            for (int i = -radius; i <= radius; i++) {
                for (int j = -radius; j <= radius; j++) {
                    scale += src.at<uchar>(row + i, col + j) * kernelX[i + radius][j + radius];
                }
            }
            gradX.at<uchar>(row - radius, col - radius) = scale / 480;
        }
    }
    imshow("X edge detection", gradX);

    //Looping over the image with the y kernel
    //From this we get the gradient image
    for (int row = radius; row < src.rows - radius; row++) {
        for (int col = radius; col < src.cols - radius; col++) {
            int scale = 0;

            for (int i = -radius; i <= radius; i++) {
                for (int j = -radius; j <= radius; j++) {
                    scale += src.at<uchar>(row + i, col + j)* kernelY[i + radius][j + radius];
                }
            }
            gradY.at<uchar>(row - radius, col - radius) = scale / 480;
        }
    }

    imshow("Y edge detection", gradY);

    //Here we calculate an approximation of the gradient at every point, using both the x and y images
    for (int row = 0; row < gradF.rows; row++) {
        for (int col = 0; col < gradF.cols; col++) {

            gradF.at<uchar>(row, col) = static_cast<uchar>(sqrt(pow(gradX.at<uchar>(row, col), 2) + pow(gradY.at<uchar>(row, col), 2)));
            //Simple threshold
            if (gradF.at<uchar>(row, col) > 1) {
                gradF.at<uchar>(row, col) = 255;
            } else {
                gradF.at<uchar>(row, col) = 0;
            }
        }
    }

    imshow("Edges", gradF);

    waitKey(0);
}

int main() {

    src= imread("/home/daniel/Documents/opencvFilters/horizont.jpg", CV_LOAD_IMAGE_GRAYSCALE);

    if (src.empty()) return -1;

    doImageProcessing();
    return 0;
}