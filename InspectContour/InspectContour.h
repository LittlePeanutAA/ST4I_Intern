#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/core/ocl.hpp>
#include "svbase_include.h"


using namespace std;
using namespace cv;

namespace sv {
    /// <summary>
    /// 
    /// </summary>
    struct PixelBin {
        /// <summary>
        /// Ánh xạ tọa độ đại diện mỗi bin - index của điểm trong list contour
        /// </summary>
        map<pair<int, int>, vector<int>> Bin;
    };
    /// <summary>
    /// class chứa thông tin sau khi tạo contour trên ảnh template
    /// </summary>
    struct TrainTemplateData {
        /// <summary>
        /// Danh sách tọa độ các điểm trên contour template so với tọa độ ảnh template
        /// </summary>
        vector<cv::Point2f> TemplateContour;
        /// <summary>
        /// 
        /// </summary>
        /// <param name="template_contour"></param>
        /// <param name="template_img_size"></param>
        TrainTemplateData(std::vector<cv::Point2f> templateContour) {
            this->TemplateContour = templateContour;
        }
        /// <summary>
        /// 
        /// </summary>
        TrainTemplateData() {

        }
        // 
    };
    /// <summary>
    /// Class lưu lại thông tin điểm bị lỗi
    /// </summary>
    struct ErrorPoint
    {
        /// <summary>
        /// Tọa độ vị trí bị lỗi
        /// </summary>
        cv::Point2f Point;
        /// <summary>
        /// Khoảng cách (L1,L2) 
        /// </summary>
        double Distance;
        /// <summary>
        /// 
        /// </summary>
        /// <param name="point"></param>
        /// <param name="distance"></param>
        ErrorPoint(cv::Point2f point, double distance) {
            this->Point = point;
            this->Distance = distance;
        }
    };
    /// <summary>
    /// Kết quả so sánh hai đường contour
    /// </summary>
    struct InspectContourResult {
        /// <summary>
        /// Tọa độ các điểm trên contour ảnh cần tìm
        /// </summary>
        vector<cv::Point2f> TargetContours;
        /// <summary>
        /// Chứa thông tin điểm lỗi trên target contour khi so sánh với template contour
        /// </summary>
        vector<ErrorPoint> TargetErrorPoints;
        /// <summary>
        /// Chứa thông tin điểm lỗi trên template contour khi so sánh với target contour, template đã chuyển vè tọa độ ảnh 
        /// </summary>
        vector<ErrorPoint> TemplateErrorPoints;
        /// <summary>
        /// 
        /// </summary>
        /// <param name="converted_template_contour"></param>
        /// <param name="target_contour"></param>
        /// <param name="temp_output"></param>
        /// <param name="targ_output"></param>
        InspectContourResult(vector<cv::Point2f> targetContour, vector<ErrorPoint> targetErrorPoints, vector<ErrorPoint> templateErrorPoints) {
            this->TargetContours = targetContour;
            this->TargetErrorPoints = targetErrorPoints;
            this->TemplateErrorPoints = templateErrorPoints;
        }
        /// <summary>
        /// 
        /// </summary>
        InspectContourResult() {

        }
    };

    /// <summary>
    /// Thông số cài đặt cho InspecContour
    /// </summary>
    struct InspectContourOptions {
        /// <summary>
        /// Bật/tắt chế độ subpixel
        /// </summary>
        bool SubPixelMode;
        /// <summary>
        /// Ngưỡng dưới bộ lọc canny
        /// </summary>
        double LowerThresh;
        /// <summary>
        /// Ngưỡng trên bộ lọc canny
        /// </summary>
        double UpperThresh;
        /// <summary>
        /// 
        /// </summary>
        double MinLengthContour;
        /// <summary>
        /// Khoảng cách tối đa cho phép giữa 2 điểm tương ứng của 2 contour cần compare với nhau
        /// </summary>
        double DistanceThresh;
    };
    /// <summary>
    /// Hàm train ảnh template từ ảnh đầu vào và vùng ROI 
    /// </summary>
    /// <param name="temp_img"></param>
    /// <param name="min_length_contour"></param>
    /// <param name="option"></param>
    /// <param name="subPixelMode"></param>
    /// <returns></returns>
    CVAPI(TrainTemplateData) trainTemplate(cv::Mat inputImage, cv::RotatedRect roi, InspectContourOptions options);


    /// <summary>
    /// 
    /// </summary>
    /// <param name="temp_contour">:danh sách tọa độ contour trên tọa độ ảnh đầu vào</param>
    /// <param name="temp_size">:kích thước ảnh template crop ??</param>
    /// <param name="targ_img">:ảnh đầu vào tìm kiếm</param>
    /// <param name="rect">:chứa thông tin 4 đỉnh của ROI -> Cần chuyển về RotatedRect</param>
    /// <param name="distance_thresh">ngưỡng distance so sánh</param>
    /// <param name="min_length_contour">:</param>
    /// <param name="option">:</param>
    /// <param name="subPixelMode"></param>
    /// <returns></returns>
    CVAPI(InspectContourResult) inspectContour(cv::Mat inputImg, cv::RotatedRect roi, TrainTemplateData trainedData, InspectContourOptions options);

}


