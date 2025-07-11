#pragma once
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <vector>
#include <string>

struct CenterPrior {
    int x;
    int y;
    int stride;
};

struct Box {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
};

struct DetectionResult {
    int class_id;
    float confidence;
    cv::Rect box;
};

class ObjectDetector {
public:
    enum ModelType {
        YOLOv8,
        NanoDet
    };
public:
    ObjectDetector(const std::string& model_xml, const std::string& model_bin, 
                  const std::string& device="CPU", ModelType type=YOLOv8);
    
    std::vector<DetectionResult> detect(cv::Mat& frame);
    
private:
    void preprocessYOLOv8(const cv::Mat& image, cv::Mat& blob, float& scale, int& nw, int& nh);
    void preprocessNanoDet(const cv::Mat& image, cv::Mat& blob, float& scale, int& nw, int& nh);
    std::vector<DetectionResult> postprocessYOLOv8(const ov::Tensor& output_tensor, float scale, 
                                                  int nw, int nh, int image_width, int image_height);
    std::vector<DetectionResult> postprocessNanoDet(const ov::Tensor& output_tensor, float scale,
                                                  int nw, int nh, int image_width, int image_height);
    
    ov::Core ie;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    
    std::vector<std::string> class_names = {"volleyball"};
    cv::Scalar colors = cv::Scalar(0, 0, 255);
    
    int input_width;
    int input_height;
    ModelType model_type;
    int num_class_ = 1; // 只有排球一个类别
};
