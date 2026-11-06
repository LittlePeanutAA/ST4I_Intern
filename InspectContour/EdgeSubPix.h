#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

/// <summary>
/// 
/// </summary>
struct Contour
{
    /// <summary>
    /// 
    /// </summary>
    std::vector<cv::Point2f> points;
    /// <summary>
    /// 
    /// </summary>
    std::vector<float> direction;
    /// <summary>
    /// 
    /// </summary>
    std::vector<float> response;
};
/// <summary>
/// ?????
/// </summary>
/// <param name="gray"></param>
/// <param name="alpha"></param>
/// <param name="low"></param>
/// <param name="high"></param>
/// <param name="contours"></param>
/// <param name="hierarchy"></param>
/// <param name="mode"></param>
void EdgesSubPix(cv::Mat& gray, double alpha, int low, int high, std::vector<Contour>& contours, cv::OutputArray hierarchy, int mode);
   
/// <summary>
/// ?????
/// </summary>
/// <param name="gray"></param>
/// <param name="alpha"></param>
/// <param name="low"></param>
/// <param name="high"></param>
/// <param name="contours"></param>
void EdgesSubPix(cv::Mat& gray, double alpha, int low, int high, std::vector<Contour>& contours);
   
/// <summary>
/// ??????
/// </summary>
/// <param name="gray"></param>
/// <param name="contour"></param>
/// <param name="alpha"></param>
/// <returns></returns>
vector<Point2f> InPix2SubPix(Mat& gray, vector<vector<Point>> contour, double alpha);

