#include "pch.h"
#include <iostream>
#include "InspectContour.h"
#include "DistanceSimd.h"
#include "EdgeSubPix.h"


namespace sv {
    /// <summary>
/// Hàm tìm danh sách các tọa độ các điểm contour so với hệ tọa độ ảnh truyền vào
/// </summary>
/// <param name="image"></param>
/// <param name="min_length_contour"></param>
/// <param name=""></param>
/// <returns></returns>
    std::vector<cv::Point2f> extractContour(cv::Mat image, InspectContourOptions options) {
        return templateContour;
    }

    /// <summary>
    /// Hàm tìm danh sách các tọa độ các điểm contour từ vùng ROI so với hệ tọa độ ảnh truyền vào
    /// </summary>
    /// <param name="image"></param>
    /// <param name="roi"></param>
    /// <param name="options"></param>
    /// <returns></returns>
    std::vector<cv::Point2f> extractContourFromRoi(cv::Mat inputImage, cv::RotatedRect roi, InspectContourOptions options) {
            return cnt;           
        }
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="srcPoints"></param>
    /// <param name="center"></param>
    /// <param name="angle"></param>
    /// <param name="offsetPoints"></param>
    /// <returns></returns>
    vector<Point2f> convertCoordinate(vector<Point2f> srcPoints, Point2f center, float angle, Point2f offsetPoints) {
        vector<cv::Point2f> dstPoints;
        for (int i = 0; i < srcPoints.size(); i++) {
            cv::Point2f pointInImgCoord = sv::rotatePoint(srcPoints[i], center, angle, offsetPoints);
            dstPoints.push_back(pointInImgCoord);
        }
        return dstPoints;
    }

    /// <summary>
    /// Duyệt từng tọa độ contour và chia mỗi tọa độ vào các bin, các điểm tring cùng lân cận sẽ được nhóm vào một bin
    /// </summary>
    /// <param name="contour"></param>
    /// <param name="stride">: ???</param>
    /// <returns></returns>
    PixelBin devideBin(vector<cv::Point2f> contour, double stride) {
        // Divide points in contour into bins
        PixelBin pibin;
        for (int idx = 0; idx < contour.size(); ++idx) {
            auto& point = contour[idx];
            // Nếu cùng lân cận <stride> thì các điểm sẽ có cùng giá trị cặp pair<int,int> 
            (pibin.Bin[make_pair((int)(point.x / stride), (int)(point.y / stride))]).push_back(idx);
        }
        return pibin;
    }

    /// <summary>
    /// Hàm train ảnh template từ ảnh lớn đầu vào và vùng ROI trên giao diện gửi xuống
    /// </summary>
    /// <param name="templateImage"></param>
    /// <param name="options"></param>
    /// <returns></returns>
    TrainTemplateData trainTemplate(cv::Mat inputImage, cv::RotatedRect roi, InspectContourOptions options) {
        return TrainTemplateData(extractContourFromRoi(inputImage, roi, options));
    }

    /// <summary>
    /// Calculate distance of 2 contour
    /// </summary>
    /// <param name="cnt_src"></param>
    /// <param name="cnt_dst"></param>
    /// <param name="bin_dst"></param>
    /// <param name="distance_thresh"></param>
    /// <returns></returns>
    vector<ErrorPoint> calDisOfContour(std::vector<cv::Point2f> cnt_src, std::vector<cv::Point2f> cnt_dst, PixelBin bin_dst, double distance_thresh) {
        return output;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="cnt_src"></param>     
    /// <param name="cnt_dst"></param>
    /// <param name="bin_dst"></param>
    /// <param name="distance_thresh"></param>
    /// <returns></returns>
    vector<ErrorPoint> calDisOfContourSse2(vector<cv::Point2f> srcContour, vector<cv::Point2f> dstContour, PixelBin bin_dst, double distanceThresh) {
        return output;
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="temp_contour"></param>
    /// <param name="temp_size"></param>
    /// <param name="targ_img"></param>
    /// <param name="rect"></param>
    /// <param name="distance_thresh"></param>
    /// <param name="min_length_contour"></param>
    /// <param name="option"></param>
    /// <param name="subPixelMode"></param>
    /// <returns></returns>
    InspectContourResult inspectContour(cv::Mat inputImage, cv::RotatedRect roi, TrainTemplateData trainedData, InspectContourOptions options) {
        return result;
    }
}

