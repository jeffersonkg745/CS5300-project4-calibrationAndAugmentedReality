/**
 * Kaelyn Jefferson
 * CS5300 Project 3
 * Main that calls object recognition functions used for project 3.
 */

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;

/**
 * @brief Main function starts video/photo input and listens for user key functions in the Object recognition system.
 *
 * @param argc 2 values
 * @param argv "cs5300-project3 photo", "cs5300-project3 video"
 * @return int
 */
int main(int argc, const char *argv[])
{

    // does OR techniques on live video
    if (std::string(argv[1]) == ("video"))
    {
        cv::VideoCapture *capdev;
        cv::Mat dst;

        // capture video frame
        capdev = new cv::VideoCapture(0);
        if (!capdev->isOpened())
        {
            printf("Unable to open video for you\n");
            return (-1);
        }

        // get the properties of the image
        cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                      (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));

        cv::namedWindow("Video", 1);
        cv::Mat frame;
        int k = 0;
        int setOnce = 0;
        std::string objectLabel;

        for (;;)
        {
            if (k == 0)
            {
                delete capdev;
                capdev = new cv::VideoCapture(0);
                // resetDistanceMetrics();
                // setOnce = 0;
                /// currentFeatures = "";
                // objectLabel = "";
                // pNearestNeighbors = "";
            }
            *capdev >> frame;

            cv::imshow("Video", frame);
            char key = cv::waitKey(10);

            if (key == 'q')
            {
                break;
            }
        }
        delete capdev;
        return 0;
    }

    if (std::string(argv[1]) == ("photo"))
    {
        cv::Mat img = imread("/Users/kaelynjefferson/Documents/NEU/MSCS/MSCS semesters/2022 Spring/CS5300-project4-calibrationAndAugmentedReality/checkerboard.png", cv::IMREAD_COLOR);
        cv::Mat dst;
        std::string currentFeatures;

        if (img.empty())
        {
            std::cout << "could not read the image!" << std::endl;
            return 1;
        }
        imshow("Display window", img);
        int k = cv::waitKey(0);

        while (k != 'q')
        {

            if (k == 's') // save the image for testing purposes
            {
                imwrite("/Users/kaelynjefferson/Desktop/sampleImageThresholded.jpg", dst);
            }

            imshow("Display window", dst);
            k = cv::waitKey(0);
        }
    }
    return 0;
}