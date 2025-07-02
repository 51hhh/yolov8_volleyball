#pragma once
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <vector>
#include <string>

struct DetectionResult {
    int class_id;
    float confidence;
    cv::Rect box;
};

class YOLOv8Detector {
public:
    YOLOv8Detector(const std::string& model_xml, const std::string& model_bin, const std::string& device="CPU");
    
    std::vector<DetectionResult> detect(cv::Mat& frame);
    
private:
    void preprocess(const cv::Mat& image, cv::Mat& blob, float& scale, int& nw, int& nh);
    std::vector<DetectionResult> postprocess(const ov::Tensor& output_tensor, float scale, 
                                            int nw, int nh, int image_width, int image_height);
    
    ov::Core ie;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    
    std::vector<std::string> class_names = {"volleyball"};
    cv::Scalar colors = cv::Scalar(0, 0, 255);
    
    int input_width;
    int input_height;
};
