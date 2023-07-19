#ifndef FEATURE_EXACTION_YOLO_H
#define FEATURE_EXACTION_YOLO_H
#include <opencv2/opencv.hpp>

namespace yolo_feature
{
class yolo
{   
    public:
    const float INPUT_WIDTH = 640.0;
    const float INPUT_HEIGHT = 640.0;
    const float SCORE_THRESHOLD = 0.2;
    const float NMS_THRESHOLD = 0.4;
    const float CONFIDENCE_THRESHOLD = 0.4;
    const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};
    struct Detection
    {
        int class_id;
        float confidence;
        cv::Rect box;
    };

    public:
    std::vector<std::string> load_class_list();
    void load_net(cv::dnn::Net &net, bool is_cuda);
    cv::Mat format_yolov5(const cv::Mat &source);
    void detect(cv::Mat &image, cv::dnn::Net &net, std::vector<Detection> &output, const std::vector<std::string> &className);
};

}
#endif //FEATURE_EXACTION_YOLO_H