#include "detect.hpp"
#include <opencv2/dnn.hpp>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>

static std::string g_model_type = "fp32";

YOLOv8Detector::YOLOv8Detector(const std::string& model_xml, const std::string& model_bin, const std::string& device) {
    // 加载模型
    auto model = ie.read_model(model_xml, model_bin);
    compiled_model = ie.compile_model(model, device);
    infer_request = compiled_model.create_infer_request();
    // 读取model_type
    std::ifstream config_file("config.json");
    if (config_file) {
        nlohmann::json config;
        config_file >> config;
        if (config.contains("model_type"))
            g_model_type = config["model_type"];
    }
    // 打印所有输出节点信息
    std::cout << "[OpenVINO] 输出节点信息:" << std::endl;
    auto outputs = compiled_model.outputs();
    for (size_t i = 0; i < outputs.size(); ++i) {
        std::cout << "  输出节点 " << i << ": ";
        auto names = outputs[i].get_names();
        if (!names.empty()) {
            std::cout << "name=" << *names.begin() << ", ";
        } else {
            std::cout << "name=<无>, ";
        }
        std::cout << "shape=";
        auto shape = outputs[i].get_partial_shape();
        for (auto s : shape) std::cout << s << " ";
        std::cout << std::endl;
    }
    // 只在fp32下获取静态shape
    if (g_model_type != "int8") {
        auto input_port = compiled_model.input();
        auto input_shape = input_port.get_shape();
        input_height = input_shape[2];
        input_width = input_shape[3];
    }
}

std::vector<DetectionResult> YOLOv8Detector::detect(cv::Mat& frame) {
    static cv::Mat blob; // 复用blob，避免每帧分配
    float scale;
    int nw, nh;
    preprocess(frame, blob, scale, nw, nh);
    
    // 创建输入tensor
    auto input_tensor = infer_request.get_input_tensor();
    float* input_data = input_tensor.data<float>();
    memcpy(input_data, blob.ptr<float>(), blob.total() * blob.elemSize());
    
    // // 异步推理
    // infer_request.start_async();
    // infer_request.wait();
    // 同步推理，保证最低延迟
    infer_request.infer();
    auto output_tensor = infer_request.get_output_tensor();
    // int8模型推理后再获取shape
    if (g_model_type == "int8") {
        auto input_shape = input_tensor.get_shape();
        input_height = input_shape[2];
        input_width = input_shape[3];
    }
    return postprocess(output_tensor, scale, nw, nh, frame.cols, frame.rows);
}

void YOLOv8Detector::preprocess(const cv::Mat& image, cv::Mat& blob, float& scale, int& nw, int& nh) {
    // 与Python完全一致的预处理
    int h = image.rows;
    int w = image.cols;
    scale = std::min(static_cast<float>(input_width) / w, 
                    static_cast<float>(input_height) / h);
    nw = static_cast<int>(w * scale);
    nh = static_cast<int>(h * scale);
    
    // 保持长宽比resize
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(nw, nh));
    
    // 添加padding (与Python完全一致)
    int dw = (input_width - nw) / 2;
    int dh = (input_height - nh) / 2;
    cv::Mat padded = cv::Mat::zeros(input_height, input_width, CV_8UC3);
    padded.setTo(cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(dw, dh, nw, nh)));
    
    // 转换为模型输入格式 (参数与Python完全一致)
    cv::dnn::blobFromImage(padded, blob, 1/255.0, cv::Size(), cv::Scalar(0, 0, 0), true, true);
}

std::vector<DetectionResult> YOLOv8Detector::postprocess(const ov::Tensor& output_tensor, float scale, 
                                                        int nw, int nh, int image_width, int image_height) {
    // output_tensor shape: (1, 5, 8400)
    // 其中5表示每个anchor的5个属性（中心x, 中心y, 宽, 高, 置信度），8400为anchor数量
    const float* output_data = output_tensor.data<const float>();
    int num_anchors = output_tensor.get_shape()[2]; // 8400
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    for (int i = 0; i < num_anchors; ++i) {
        float confidence = output_data[4 * num_anchors + i];
        if (confidence > 0.5f) {
            float cx = output_data[0 * num_anchors + i];
            float cy = output_data[1 * num_anchors + i];
            float w  = output_data[2 * num_anchors + i];
            float h  = output_data[3 * num_anchors + i];
            int dw = (input_width - nw) / 2;
            int dh = (input_height - nh) / 2;
            int x1 = static_cast<int>((cx - w/2 - dw) / scale);
            int y1 = static_cast<int>((cy - h/2 - dh) / scale);
            int x2 = static_cast<int>((cx + w/2 - dw) / scale);
            int y2 = static_cast<int>((cy + h/2 - dh) / scale);
            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            x2 = std::min(image_width - 1, x2);
            y2 = std::min(image_height - 1, y2);
            boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
            scores.push_back(confidence);
            class_ids.push_back(0);
        }
    }
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, 0.5f, 0.5f, indices);
    std::vector<DetectionResult> final_results;
    for (int idx : indices) {
        DetectionResult det;
        det.class_id = class_ids[idx];
        det.confidence = scores[idx];
        det.box = boxes[idx];
        final_results.push_back(det);
    }
    return final_results;
}
