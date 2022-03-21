/**
 * Kaelyn Jefferson
 * CS5300 Project 3
 * Object recognition functions used for project 3.
 */
#include "CalibrationAndARfxns.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>
#include <iostream>
#include <vector>

/**
 * @brief Question 1: Detect and extract the chessboard corners (using checkerboard.png).
 *
 * @param src cv::Mat type image
 * @param dst cv::Mat type image
 * @return int
 * https://github.com/opencv/opencv/blob/4.x/samples/cpp/tutorial_code/calib3d/camera_calibration/camera_calibration.cpp
 */
int detectAndExtractCorners(cv::Mat &src, cv::Mat &dst, int num, std::vector<std::vector<cv::Vec3f>> &point_list, std::vector<std::vector<cv::Point2f>> &corner_list)
{
    // use for question 2
    std::vector<cv::Vec3f> point_set;

    cv::Size pattern_size(9, 6);

    std::vector<cv::Point2f> corner_set;
    bool pattern_found = findChessboardCorners(src, pattern_size, corner_set, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);

    if (pattern_found)
    {
        cv::Mat gray;
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

        cv::cornerSubPix(gray, corner_set, cv::Size(19, 13), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::Type::MAX_ITER + cv::TermCriteria::Type::EPS, 30, 0.1));
    }

    cv::drawChessboardCorners(dst, pattern_size, cv::Mat(corner_set), pattern_found);

    // save new calibration image for problem 2
    if (num == 1)
    {
        corner_list.push_back(corner_set);

        // measure the "real world units" by counting units of checkerboard squares
        // for (int i = 0; i < corner_set.size(); i++)
        // {
        //     std::cout << "corner set: x=" << corner_set[i].x << " y=" << corner_set[i].y << std::endl;
        // }

        int xCoord = 0;
        int zCoord = 0;

        for (int i = 0; i < corner_set.size(); i++)
        {

            cv::Vec3f currentCorner = {0, 0, 0};

            currentCorner[0] = xCoord;
            currentCorner[1] = zCoord;
            currentCorner[2] = 0; // y is always zero in our case?

            // add points to point list
            point_set.push_back(currentCorner);

            // std::cout << "corner set: x=" << corner_set[i].x << " y=" << corner_set[i].y << std::endl;
            // std::cout << "our corner in real coord: x=" << currentCorner[0] << "z=" << currentCorner[1] << "y=" << currentCorner[2] << std::endl;

            if (xCoord == 9)
            {
                xCoord = 0;
                zCoord += 1;
                continue;
            }

            xCoord += 1;
        }

        // add the point set to the point list
        point_list.push_back(point_set);

        // error checking here
        if (point_set.size() != corner_set.size())
        {
            std::cout << "Point set and corner set should have same number of values." << std::endl;
        }

        if (point_list.size() != corner_list.size())
        {
            std::cout << "Point list and corner list should have same number of values." << std::endl;
        }
    }

    return 0;
}
