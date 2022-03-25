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
using namespace cv;

/**
 * @brief Question 1: Detect and extract the chessboard corners (using checkerboard.png).
 *
 * @param src cv::Mat type image
 * @param dst cv::Mat type image
 * @param int num: 0: doesn't draw corners; 1: adds to corner list; 2: draws corners
 * @return int
 * https://github.com/opencv/opencv/blob/4.x/samples/cpp/tutorial_code/calib3d/camera_calibration/camera_calibration.cpp
 */
int detectAndExtractCorners(bool isCheckerboard, cv::Mat &src, cv::Mat &dst, int num, std::vector<std::vector<cv::Vec3f>> &point_list, std::vector<std::vector<cv::Point2f>> &corner_list)
{
    std::vector<cv::Vec3f> point_set;
    std::vector<cv::Point2f> corner_set; // note for circles, this is the "center_set"

    if (isCheckerboard)
    {
        cv::Size pattern_size(9, 6);
        bool pattern_found = findChessboardCorners(src, pattern_size, corner_set, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
        if (pattern_found)
        {
            cv::Mat gray;
            cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
            cv::cornerSubPix(gray, corner_set, cv::Size(19, 13), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::Type::MAX_ITER + cv::TermCriteria::Type::EPS, 30, 0.1));
            cv::drawChessboardCorners(dst, pattern_size, cv::Mat(corner_set), pattern_found);
        }
    }

    if (!isCheckerboard)
    {
        Size pattern_size(7, 7);
        bool circle_pattern_found = findCirclesGrid(src, pattern_size, corner_set);
        if (circle_pattern_found)
        {
            drawChessboardCorners(dst, pattern_size, Mat(corner_set), circle_pattern_found);
        }
    }

    // save new calibration image for task 2
    if (num >= 1)
    {
        corner_list.push_back(corner_set);
        int xCoord = 0;
        int yCoord = 0;
        for (int i = 0; i < corner_set.size(); i++)
        {
            cv::Vec3f currentCorner = {0, 0, 0}; // {x,y,z}
            currentCorner[0] = xCoord;           // x in our case increases going from left to right on the board
            currentCorner[1] = yCoord;           // y in our case increases as go top to bottom on the board
            currentCorner[2] = 0;                // z is always zero in our case since it points out of the board (+z is pointing away from board)

            // add points to point list
            point_set.push_back(currentCorner);
            std::cout << currentCorner << std::endl;

            if (isCheckerboard && xCoord == 8)
            {
                xCoord = 0;
                yCoord += 1;
                continue;
            }
            if (!isCheckerboard && xCoord == 6)
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

    // print the number of corners found
    std::cout << "Number of corners found: " << corner_set.size() << std::endl;
    std::cout << "first corner shown here: " << corner_set[0] << std::endl;

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
            std::string s = std::to_string(camera_matrix.at<cv::Vec2f>(i, j)[0]) + ", " + std::to_string(camera_matrix.at<cv::Vec2f>(i, j)[1]);
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

    cv::Size image_size(720, 1280);
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

    // std::cout << point_list[0].size() << std::endl;
    // std::cout << corner_list[0].size() << std::endl;

    // WORKS FOR SQUARE
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

std::string getLastLineOfFile(std::string fileName)
{

    std::ifstream myfile;
    myfile.open(fileName);
    std::string myline;
    std::string s = "";

    if (myfile.is_open())
    {
        while (myfile)
        {
            std::getline(myfile, myline);

            // keep adding to this string so will end on the last line
            if (myline.size() != 0)
            {
                s = myline;
            }
        }
    }
    myfile.close();

    return s;
}

int getMatrixFromString(std::string matrixString, cv::Mat &camera_matrix)
{
    cv::Size matrix_size(3, 3);
    camera_matrix.create(matrix_size, CV_64FC1);

    std::cout << matrixString << std::endl;

    std::string str = matrixString;
    int i = 0;
    int j = 0;
    while (str.length() > 5)
    {

        cv::Vec2f vec = {0, 0};

        std::string matrixPart = str.substr(0, str.find(",") + 1);
        std::remove(matrixPart.begin(), matrixPart.end(), ' ');
        vec[0] = atof(matrixPart.c_str());
        str.erase(0, str.find(",") + 1);

        // std::cout << vec[0] << std::endl;

        std::string matrixPart2 = str.substr(0, str.find(",") + 1);
        std::remove(matrixPart.begin(), matrixPart.end(), ' ');
        vec[1] = atof(matrixPart2.c_str());
        str.erase(0, str.find(",") + 1);

        // camera_matrix.at<cv::Vec2f>(i, j)[0] = vec[0];
        // camera_matrix.at<cv::Vec2f>(i, j)[1] = vec[1];

        camera_matrix.at<cv::Vec2f>(i, j) = vec;
        std::cout << camera_matrix.at<cv::Vec2f>(i, j) << std::endl;

        // std::cout << vec[1] << std::endl;

        i += 1;
        if (i == 3)
        {
            i = 0;
            j += 1;
        }
    }

    return 0;
}

int calcPosOfCamera(std::vector<std::vector<cv::Vec3f>> point_list, std::vector<std::vector<cv::Point2f>> corner_list, cv::Mat &rvec,
                    cv::Mat &tvec, cv::Mat &camera_matrix, std::vector<float> &dist_coefficients)
{

    // std::vector<float> dist_coefficients;
    //  get points and corners of the frame that was detected with the board
    std::vector<cv::Vec3f> objectPoints = point_list[point_list.size() - 1];
    std::vector<cv::Point2f> imagePoints = corner_list[corner_list.size() - 1];

    // http://www.learningaboutelectronics.com/Articles/How-to-count-the-number-of-lines-in-a-file-C++.php
    std::string matrixString = getLastLineOfFile("CameraMatrix.txt");

    // format strings into appropriate matrix and vector data structures

    // getMatrixFromString(matrixString, camera_matrix); //instead of calling, put below for now

    // TODO: FIX PROBLEM WITH HELPER FXN
    cv::Size matrix_size(3, 3);
    camera_matrix.create(matrix_size, CV_64FC1);
    // std::cout << matrixString << std::endl;
    std::string str = matrixString;
    int i = 0;
    int j = 0;
    while (str.length() > 5)
    {
        cv::Vec2f vec = {0, 0};
        std::string matrixPart = str.substr(0, str.find(",") + 1);
        std::remove(matrixPart.begin(), matrixPart.end(), ' ');
        vec[0] = atof(matrixPart.c_str());
        str.erase(0, str.find(",") + 1);
        std::string matrixPart2 = str.substr(0, str.find(",") + 1);
        std::remove(matrixPart.begin(), matrixPart.end(), ' ');
        vec[1] = atof(matrixPart2.c_str());
        str.erase(0, str.find(",") + 1);
        camera_matrix.at<cv::Vec2f>(i, j) = vec;
        // std::cout << camera_matrix.at<cv::Vec2f>(i, j) << std::endl;

        i += 1;
        if (i == 3)
        {
            i = 0;
            j += 1;
        }
    }

    if (dist_coefficients.size() == 0) // don't want to resave this for every time it scans through
    {

        // get the vector from the last line in the distortion coefficients
        cv::Mat dist_coeff;
        std::string distString = getLastLineOfFile("DistortionCoefficients.txt");
        std::remove(distString.begin(), distString.end(), ' ');

        int z = 0;
        while (distString.length() > 6)
        {
            std::string num = distString.substr(0, distString.find(","));
            float numFloat = std::stof(num);
            dist_coefficients.push_back(numFloat);
            // dist_coeff.at<cv::Vec2f>(0, z) = numFloat;
            distString.erase(0, distString.find(",") + 1);
            z += 1;
        }
    }

    /*
        for (int i = 0; i < dist_coefficients.size(); i++)
        {
            std::cout << dist_coefficients[i] << std::endl;
        }
        */

    // std::vector<float> rvec;
    // std::vector<float> tvec;

    cv::solvePnP(objectPoints, imagePoints, camera_matrix, dist_coefficients, rvec, tvec);

    // std::cout << rvec.rows << std::endl; //3
    // std::cout << rvec.cols << std::endl; //1
    // std::cout << tvec.rows << std::endl;
    // std::cout << tvec.cols << std::endl;

    return 0;
}

// delete this fxn
int detectCornersHarrisFxn(cv::Mat &frame, int num, std::vector<std::vector<cv::Vec3f>> &point_list, std::vector<std::vector<cv::Point2f>> &corner_list)
{
    cv::Mat dst = Mat::zeros(frame.size(), CV_32FC1);
    cv::Mat src_gray;
    cv::cvtColor(frame, src_gray, cv::COLOR_BGR2GRAY);
    cv::cornerHarris(src_gray, dst, 2, 3, 0.04);

    // below implementation sourced from: https://docs.opencv.org/3.4/d4/d7d/tutorial_harris_detector.html
    // explanation on edge detection problem: https://anothertechs.com/programming/cpp/opencv-corner-harris-cpp/
    Mat dst_norm, dst_norm_scaled;
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(dst_norm, dst_norm_scaled);
    int thresh = 120;
    // int thresh = 130; //use 130 for extension
    int max_thresh = 255;

    std::vector<cv::Point2f> corner_set;
    std::vector<cv::Vec3f> point_set;

    for (int i = 0; i < dst_norm.rows - 100; i++) // cut off bottom portion b/c of app logo detecting corners
    {
        for (int j = 0; j < dst_norm.cols; j++)
        {
            if ((int)dst_norm.at<float>(i, j) > thresh) // checks against the threshold to see if edge
            {

                // save the points found in the corner_set
                cv::Point2f pointToAdd = Point(j, i);

                if (corner_set.size() == 0)
                {
                    corner_set.push_back(pointToAdd);
                    std::cout << pointToAdd << std::endl;
                    circle(dst_norm_scaled, Point(j, i), 5, Scalar(0), 2, 8, 0);
                    continue;
                }

                // if its different enough then add to corner__set
                float xDiff = 100;
                float yDiff = 100;

                // check to see if we already have that corner
                for (int k = 0; k < corner_set.size(); k++)
                {

                    cv::Point2f pointToCompare = corner_set[k];

                    // get the smallest difference between all the points
                    float x = abs(pointToAdd.x - pointToCompare.x);
                    float y = abs(pointToAdd.y - pointToCompare.y);

                    // get the smallest difference then compare after compare all
                    if (x < xDiff)
                    {
                        xDiff = x;
                    }
                    if (y < yDiff)
                    {
                        yDiff = y;
                    }
                }

                // look at smallest difference of the point we want to add,
                // if its a bigger difference than ones existing, then add
                if (xDiff > 5 || yDiff > 5)
                {
                    corner_set.push_back(pointToAdd);
                    std::cout << pointToAdd << std::endl;
                    circle(dst_norm_scaled, Point(j, i), 5, Scalar(0), 2, 8, 0);
                }
            }
        }
    }

    std::cout << corner_set.size() << std::endl;
    frame = dst_norm_scaled;

    // once we find the corner_set, refine it by using:
    // cv::Mat gray;
    // cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // already converted to gray frame above
    cv::cornerSubPix(dst_norm_scaled, corner_set, cv::Size(19, 13), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::Type::MAX_ITER + cv::TermCriteria::Type::EPS, 30, 0.1));

    // getting the real world coordinates
    if (num >= 1)
    {
        corner_list.push_back(corner_set);

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

            if (xCoord == 5) // not as clear as the checkerboard (but puts on different rows )
            {                // not symmetrical due to patter but has 1x4 on first row and 1x3 on second
                             // unless it detects 8 corners (but shows consistent 7)
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
    cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);

    return 0;
}
