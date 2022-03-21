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

/**
 * @brief Question 1: Detect and extract the chessboard corners (using checkerboard.png).
 *
 * @param src cv::Mat type image
 * @param dst cv::Mat type image
 * @return int
 * https://github.com/opencv/opencv/blob/4.x/samples/cpp/tutorial_code/calib3d/camera_calibration/camera_calibration.cpp
 */
int detectAndExtractCorners(cv::Mat &src, cv::Mat &dst)
{
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
    // std::cout << "num of corners: " << corner_set[1].x << std::endl;

    return 0;
}
