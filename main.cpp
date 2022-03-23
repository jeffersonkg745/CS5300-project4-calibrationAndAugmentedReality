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
#include "CalibrationAndARfxns.cpp"
#include "CalibrationAndARfxns.h"
#include <opencv2/calib3d.hpp>
#include <string>
#include <cstring>
#include <fstream>
using namespace cv;

/**
 * @brief Main function starts video/photo input and listens for user key functions in the Object recognition system.
 *
 * @param argc 2 values
 * @param argv "./cs5300-project4-calibrationAndAugmentedReality video"
 * @return int
 */
int main(int argc, const char *argv[])
{

    std::vector<std::vector<cv::Vec3f>> point_list;    // keeps list of 3d pos in world coordinates of corners
    std::vector<std::vector<cv::Point2f>> corner_list; // keeps list of all corner coordinates

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

        for (;;)
        {
            if (k == 0)
            {
                delete capdev;
                capdev = new cv::VideoCapture(0);
            }
            *capdev >> frame;

            if (k == 1) // detect and extract chessboard corners (Q1)
            {
                detectAndExtractCorners(frame, frame, 0, point_list, corner_list);
            }
            else if (k == 2) // select calibration images (Q2)
            {
                // user selects calibration images and updates point_list and corner_list used in question 3
                detectAndExtractCorners(frame, frame, 1, point_list, corner_list);
                std::cout << point_list.size() << std::endl;
                std::cout << corner_list.size() << std::endl;

                // saving the image we calibrated with (will overwrite current images if they exist)
                std::string imageNum = std::to_string(point_list.size());
                std::string path = "/Users/kaelynjefferson/Documents/NEU/MSCS/MSCS semesters/2022 Spring/CS5300-project4-calibrationAndAugmentedReality/imagesUsedForCalibration/" + imageNum + ".jpg";
                imwrite(path, frame);

                k = 1;
            }
            else if (k == 3) // calibrate the camera (task 3)
            {

                // only calibrate the camera if we have 5 images
                if (corner_list.size() < 5)
                {
                    k = 1;
                }

                calibrateOurCamera(frame, point_list, corner_list);

                k = 4;
            }
            else if (k == 5) // calculate the current position of the camera (task 4)
            {
                detectAndExtractCorners(frame, frame, 1, point_list, corner_list);

                // loops through continuously trying to detec the corners until it finds them, then calc the pos of the camera
                if (point_list.size() > 0)
                {
                    calcPosOfCamera(point_list, corner_list);
                    k = 6;
                }
            }

            cv::imshow("Video", frame);
            char key = cv::waitKey(10);

            if (key == 'q')
            {
                break;
            }
            else if (key == 'd') // detect and extract chessboard corners (Q1)
            {
                k = 1;
            }
            else if (key == 's') // select calibration images (Q2)
            {
                k = 2;
            }
            else if (key == 'c') // calibrate the camera (task 3)
            {
                k = 3;
            }
            else if (key == 'p') // calculate the current position of the camera (task 4)
            {
                k = 5;
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