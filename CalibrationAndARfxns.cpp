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
#include <opencv2/calib3d.hpp>
#include <string>
#include <cstring>
#include <fstream>

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
        int yCoord = 0;

        for (int i = 0; i < corner_set.size(); i++)
        {

            cv::Vec3f currentCorner = {0, 0, 0}; // {x,y,z}

            currentCorner[0] = xCoord; // x in our case increases going from left to right on the board
            currentCorner[1] = yCoord; // y in our case increases as go top to bottom on the board
            currentCorner[2] = 0;      // z is always zero in our case since it points out of the board (+z is pointing away from board)

            // add points to point list
            point_set.push_back(currentCorner);
            // std::cout << i << std::endl;
            // std::cout << currentCorner << std::endl;

            // std::cout << "corner set: x=" << corner_set[i].x << " y=" << corner_set[i].y << std::endl;
            // std::cout << "our corner in real coord: x=" << currentCorner[0] << "z=" << currentCorner[1] << "y=" << currentCorner[2] << std::endl;

            if (xCoord == 8)
            {
                xCoord = 0;
                yCoord += 1;
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

int writeToFile(std::string fileName, std::string lineToWrite)
{
    std::ofstream fout;
    fout.open(fileName, std::ios_base::app);
    fout << lineToWrite;
    fout << std::endl;
    fout.close();

    return 0;
}

std::string printMatrix(cv::Mat camera_matrix)
{
    std::cout << "\nCamera Matrix: "
              << std::endl;

    std::string cameraMatStr = "";

    for (int i = 0; i < camera_matrix.rows; i++)
    {
        for (int j = 0; j < camera_matrix.cols; j++)
        {
            std::cout << camera_matrix.at<cv::Vec2f>(i, j) << std::endl;
            std::string s = "[" + std::to_string(camera_matrix.at<cv::Vec2f>(i, j)[0]) + ", " + std::to_string(camera_matrix.at<cv::Vec2f>(i, j)[1]) + "]";
            cameraMatStr.append(s);
            cameraMatStr.append(", ");
        }
    }

    return cameraMatStr;
}

std::string printDistCoefficients(std::vector<float> dist_coeff)
{
    std::string distortionCoefficients = "";
    std::cout << "\nDistortion coefficients:"
              << std::endl;
    for (int k = 0; k < dist_coeff.size(); k++)
    {
        std::cout << dist_coeff[k] << std::endl;
        std::string s = std::to_string(dist_coeff[k]);
        distortionCoefficients.append(s);
        distortionCoefficients.append(", ");
    }

    return distortionCoefficients;
}

int calibrateOurCamera(cv::Mat &frame, std::vector<std::vector<cv::Vec3f>> &point_list, std::vector<std::vector<cv::Point2f>> &corner_list)
{

    cv::Size image_size(720, 1280); // not sure
    cv::Mat camera_matrix;
    cv::Size matrix_size(3, 3);
    camera_matrix.create(matrix_size, CV_64FC1);

    camera_matrix.at<cv::Vec2f>(0, 0) = 1;
    camera_matrix.at<cv::Vec2f>(0, 1) = 0;
    camera_matrix.at<cv::Vec2f>(0, 2) = frame.cols / 2;
    camera_matrix.at<cv::Vec2f>(1, 0) = 0;
    camera_matrix.at<cv::Vec2f>(1, 1) = 1;
    camera_matrix.at<cv::Vec2f>(1, 2) = frame.rows / 2;
    camera_matrix.at<cv::Vec2f>(2, 0) = 0;
    camera_matrix.at<cv::Vec2f>(2, 1) = 0;
    camera_matrix.at<cv::Vec2f>(2, 2) = 1;

    std::vector<float> dist_coeff;
    std::vector<cv::Mat> rotation_vec;
    std::vector<cv::Mat> translation_vec;
    double reprojection_error;

    reprojection_error = cv::calibrateCamera(point_list, corner_list, image_size, camera_matrix, dist_coeff, rotation_vec, translation_vec, cv::CALIB_FIX_ASPECT_RATIO);

    // printing the corresponding camera matrix, distortion coefficients, and reprojection error
    std::string cameraMatStr = printMatrix(camera_matrix);
    std::string distortionCoefficients = printDistCoefficients(dist_coeff);
    std::cout << "\nfinal reprojection error: " << reprojection_error << std::endl;

    // TODO: save these also to a file somewhere
    /*
    std::cout << "rotation vector: \n"
              << std::endl;
    for (int k = 0; k < rotation_vec.size(); k++)
    {
        std::cout << rotation_vec[k] << std::endl;
    }

    std::cout << "translation vector: \n"
              << std::endl;
    for (int k = 0; k < translation_vec.size(); k++)
    {
        std::cout << translation_vec[k] << std::endl;
    }
    */

    writeToFile("CameraMatrix.txt", cameraMatStr);
    writeToFile("DistortionCoefficients.txt", distortionCoefficients);

    return 0;
}
