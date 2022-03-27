/**
 * Kaelyn Jefferson
 * CS5300 Project 4
 * Main that calls calibration and augmented reality functions used for project 4.
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
 * @param argc 3 values
 * @param argv example : "./cs5300-project4-calibrationAndAugmentedReality video circle"
 * @return int
 */
int main(int argc, const char *argv[])
{

    std::vector<std::vector<cv::Vec3f>> point_list;    // keeps list of 3d pos in world coordinates of corners
    std::vector<std::vector<cv::Point2f>> corner_list; // keeps list of all corner coordinates

    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat camera_matrix;
    std::vector<float> dist_coefficients;

    // does OR techniques on live video
    if (std::string(argv[1]) == ("video"))
    {
        cv::VideoCapture *capdev;
        cv::Mat dst;

        capdev = new cv::VideoCapture(0);
        if (!capdev->isOpened())
        {
            printf("Unable to open video for you\n");
            return (-1);
        }

        cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                      (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));

        cv::namedWindow("Video", 1);
        cv::Mat frame;
        int k = 0;
        int p = 0;

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
                if (std::string(argv[2]) == ("checkerboard"))
                {
                    detectAndExtractCorners(true, frame, frame, 0, point_list, corner_list);
                }
                else if (std::string(argv[2]) == ("circle"))
                {
                    detectAndExtractCorners(false, frame, frame, 0, point_list, corner_list);
                }
            }
            else if (k == 2) // select calibration images (Q2)
            {
                if (std::string(argv[2]) == ("checkerboard"))
                {
                    // user selects calibration images and updates point_list and corner_list used in question 3
                    detectAndExtractCorners(true, frame, frame, 1, point_list, corner_list);
                    std::cout << point_list.size() << std::endl;
                    std::cout << corner_list.size() << std::endl;

                    // saving the image we calibrated with (will overwrite current images if they exist)
                    std::string imageNum = std::to_string(point_list.size());
                    std::string path = "/Users/kaelynjefferson/Documents/NEU/MSCS/MSCS semesters/2022 Spring/CS5300-project4-calibrationAndAugmentedReality/imagesUsedForCalibration/" + imageNum + ".jpg";
                    imwrite(path, frame);
                    k = 1;
                }
                else if (std::string(argv[2]) == ("circle"))
                {
                    // user selects calibration images and updates point_list and corner_list used in question 3
                    detectAndExtractCorners(false, frame, frame, 1, point_list, corner_list);
                    std::cout << point_list.size() << std::endl;
                    std::cout << corner_list.size() << std::endl;

                    // saving the image we calibrated with (will overwrite current images if they exist)
                    std::string imageNum = std::to_string(point_list.size());
                    std::string path = "/Users/kaelynjefferson/Documents/NEU/MSCS/MSCS semesters/2022 Spring/CS5300-project4-calibrationAndAugmentedReality/imagesUsedForCalibrationExt/" + imageNum + ".jpg";
                    imwrite(path, frame);
                    k = 1;
                }
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
                if (std::string(argv[2]) == ("checkerboard"))
                {
                    detectAndExtractCorners(true, frame, frame, 1, point_list, corner_list);
                }
                if (std::string(argv[2]) == ("circle"))
                {
                    detectAndExtractCorners(false, frame, frame, 1, point_list, corner_list);
                }

                // loops through continuously trying to detect the corners until it finds them, then calc the pos of the camera
                if (point_list.size() > 0)
                {

                    calcPosOfCamera(point_list, corner_list, rvec, tvec, camera_matrix, dist_coefficients);

                    // print the vectors in real time when detects the grid
                    std::cout << "\nRVEC: " << std::endl;
                    for (int i = 0; i < rvec.rows; i++)
                    {
                        for (int j = 0; j < rvec.cols; j++)
                        {
                            std::cout << rvec.at<cv::Vec2f>(i, j) << std::endl;
                        }
                    }

                    std::cout << "\nTVEC: " << std::endl;
                    for (int i = 0; i < tvec.rows; i++)
                    {
                        for (int j = 0; j < tvec.cols; j++)
                        {
                            std::cout << tvec.at<cv::Vec2f>(i, j) << std::endl;
                        }
                    }
                }
            }
            else if (k == 6) // project outside corners or 3D axes
            {

                // call these to get updated coordinates
                if (std::string(argv[2]) == ("checkerboard"))
                {
                    detectAndExtractCorners(true, frame, frame, 2, point_list, corner_list);
                }
                if (std::string(argv[2]) == ("circle"))
                {
                    detectAndExtractCorners(false, frame, frame, 2, point_list, corner_list);
                }

                if (point_list.size() > 0)
                {
                    calcPosOfCamera(point_list, corner_list, rvec, tvec, camera_matrix, dist_coefficients);
                    std::vector<cv::Point2f> imagePoints;
                    std::vector<cv::Vec3f> our_points = point_list[point_list.size() - 1];
                    cv::projectPoints(our_points, rvec, tvec, camera_matrix, dist_coefficients, imagePoints);
                    std::vector<cv::Point2f> cornerPointsOnly;

                    for (int i = 0; i < imagePoints.size(); i++)
                    {
                        std::cout << our_points[i] << std::endl;
                        if (std::string(argv[2]) == ("checkerboard")) // these are the corner points for checkerboard
                        {
                            if (our_points[i][0] == 0 && our_points[i][1] == 0 || our_points[i][0] == 8 && our_points[i][1] == 0 || our_points[i][0] == 0 && our_points[i][1] == 5 || our_points[i][0] == 8 && our_points[i][1] == 5)
                            {
                                cornerPointsOnly.push_back(imagePoints[i]);
                                std::cout << imagePoints[i] << std::endl;
                            }
                        }
                        if (std::string(argv[2]) == ("circle")) // these are the corner points for circle
                        {
                            if (our_points[i][0] == 0 && our_points[i][1] == 0 || our_points[i][0] == 6 && our_points[i][1] == 0 || our_points[i][0] == 0 && our_points[i][1] == 6 || our_points[i][0] == 6 && our_points[i][1] == 6)
                            {
                                cornerPointsOnly.push_back(imagePoints[i]);
                                std::cout << imagePoints[i] << std::endl;
                            }
                        }
                    }

                    for (int i = 0; i < cornerPointsOnly.size(); i++)
                    {
                        cv::circle(frame, cornerPointsOnly[i], 3, Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
                    }
                }
            }
            else if (k == 7) // draws the virtual object
            {
                std::vector<cv::Point2f> imagePoints;
                std::vector<cv::Vec3f> our_points = point_list[point_list.size() - 1];

                if (std::string(argv[2]) == ("checkerboard"))
                {
                    detectAndExtractCorners(true, frame, frame, 2, point_list, corner_list);
                }
                if (std::string(argv[2]) == ("circle"))
                {
                    detectAndExtractCorners(false, frame, frame, 2, point_list, corner_list);
                }

                calcPosOfCamera(point_list, corner_list, rvec, tvec, camera_matrix, dist_coefficients);
                cv::projectPoints(our_points, rvec, tvec, camera_matrix, dist_coefficients, imagePoints);

                std::vector<cv::Point2f> top;
                std::vector<cv::Point2f> right;
                std::vector<cv::Point2f> left;
                std::vector<cv::Point2f> bottom;

                for (int i = 0; i < imagePoints.size(); i++)
                {
                    if (our_points[i][0] == 0 && our_points[i][1] == 0 || our_points[i][0] == 5 && our_points[i][1] == 0)
                    {
                        top.push_back(imagePoints[i]);
                    }

                    if (our_points[i][0] == 0 && our_points[i][1] == 0 || our_points[i][0] == 0 && our_points[i][1] == 3)
                    {
                        left.push_back(imagePoints[i]);
                    }

                    if (our_points[i][0] == 0 && our_points[i][1] == 3 || our_points[i][0] == 5 && our_points[i][1] == 3)
                    {
                        bottom.push_back(imagePoints[i]);
                    }
                    if (our_points[i][0] == 5 && our_points[i][1] == 0 || our_points[i][0] == 5 && our_points[i][1] == 3)
                    {
                        right.push_back(imagePoints[i]);
                    }
                }

                // make the lines for each of the 2 coordinates in each category
                cv::line(frame, top[0], top[1], Scalar(0, 0, 255), 3, 8, 0);
                cv::line(frame, left[0], left[1], Scalar(250, 135, 206), 3, 8, 0);
                cv::line(frame, bottom[0], bottom[1], Scalar(0, 0, 255), 3, 8, 0);
                cv::line(frame, right[0], right[1], Scalar(0, 0, 255), 3, 8, 0);
            }

            cv::imshow("Video", frame);
            char key = cv::waitKey(10);

            if (key == 'q')
            {
                break;
            }
            else if (key == 'd') // detect and extract chessboard corners (task 1)
            {
                k = 1;
            }
            else if (key == 's') // select calibration images (task 2)
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
            else if (key == 'o') // project outside corners or 3d axes (task 5)
            {
                k = 6;
            }
            else if (key == 'v') // making a virtual object made out of lines (task 6)
            {
                k = 7;
            }
        }
        delete capdev;
        return 0;
    }

    if (std::string(argv[1]) == ("secondproject"))
    {
        cv::VideoCapture *capdev;
        cv::Mat dst;
        capdev = new cv::VideoCapture(0);
        if (!capdev->isOpened())
        {
            printf("Unable to open video for you\n");
            return (-1);
        }

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

            if (k == 1) // detecting robust features (Q7)
            {
                // not necessary for this problem to save new 3d coords so set parameter to 0
                detectCornersHarrisFxn(frame, 0, point_list, corner_list); // detecting robust features (task 7)
            }
            cv::imshow("Video", frame);
            char key = cv::waitKey(10);

            if (key == 'q')
            {
                break;
            }
            else if (key == 'd') // detecting robust features (Q7)
            {
                k = 1;
            }
        }
        delete capdev;
        return 0;
    }
}