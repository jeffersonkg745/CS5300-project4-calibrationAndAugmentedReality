/**
 * Kaelyn Jefferson
 * CS5300 Project 4
 * Calibration and AR functions (header file) used for project 4.
 */

#ifndef CALIBRATIONANDARFXNS_H
#define CALIBRATIONANDARFXNS_H

// Detect and extract chessboard corners(Q1)
int detectAndExtractCorners(bool isCheckerboard, cv::Mat &src, cv::Mat &dst, int num, std::vector<std::vector<cv::Vec3f>> &point_list, std::vector<std::vector<cv::Point2f>> &corner_list);

// helper functions
int writeToFile(std::string fileName, std::string lineToWrite);
std::string printMatrix(cv::Mat camera_matrix);
std::string printDistCoefficients(std::vector<float> dist_coeff);
std::string getLastLineOfFile(std::string fileName);
int getMatrixFromString(std::string matrixString, cv::Mat &camera_matrix);

// calibrates the camera
int calibrateOurCamera(cv::Mat &frame, std::vector<std::vector<cv::Vec3f>> &point_list, std::vector<std::vector<cv::Point2f>> &corner_list);

// calculates the position of the camera
int calcPosOfCamera(std::vector<std::vector<cv::Vec3f>> point_list, std::vector<std::vector<cv::Point2f>> corner_list, cv::Mat &rvec,
                    cv::Mat &tvec, cv::Mat &camera_matrix, std::vector<float> &dist_coefficients);

// uses the harris corner opencv function
int detectCornersHarrisFxn(cv::Mat &frame, int num, std::vector<std::vector<cv::Vec3f>> &point_list, std::vector<std::vector<cv::Point2f>> &corner_list);

#endif /* CALIBRATIONANDARFXNS_H */