#include "detect.hpp"
#include <opencv2/dnn.hpp>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>

static std::string g_model_type = "fp32";

ObjectDetector::ObjectDetector(const std::string& model_xml, const std::string& model_bin, 
                             const std::string& device, ModelType type) : model_type(type) {
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
    
    // 获取输入尺寸
    auto input_port = compiled_model.input();
    auto input_shape = input_port.get_partial_shape();
    input_height = input_shape[2].get_length();
    input_width = input_shape[3].get_length();
}

std::vector<DetectionResult> ObjectDetector::detect(cv::Mat& frame) {
    static cv::Mat blob; // 复用blob，避免每帧分配
    float scale;
    int nw, nh;
    
    if (model_type == NanoDet) {
        preprocessNanoDet(frame, blob, scale, nw, nh);
    } else {
        preprocessYOLOv8(frame, blob, scale, nw, nh);
    }
    
    // 创建输入tensor
    auto input_tensor = infer_request.get_input_tensor();
    float* input_data = input_tensor.data<float>();
    memcpy(input_data, blob.ptr<float>(), blob.total() * blob.elemSize());
    
    // 同步推理
    infer_request.infer();
    auto output_tensor = infer_request.get_output_tensor();
    
    // 根据模型类型选择后处理方法
    switch(model_type) {
        case YOLOv8:
            return postprocessYOLOv8(output_tensor, scale, nw, nh, frame.cols, frame.rows);
        case NanoDet:
            return postprocessNanoDet(output_tensor, scale, nw, nh, frame.cols, frame.rows);
        default:
            return {};
    }
}

void ObjectDetector::preprocessYOLOv8(const cv::Mat& image, cv::Mat& blob, float& scale, int& nw, int& nh) {
    int h = image.rows;
    int w = image.cols;
    scale = std::min(static_cast<float>(input_width) / w, 
                    static_cast<float>(input_height) / h);
    nw = static_cast<int>(w * scale);
    nh = static_cast<int>(h * scale);
    
    // 保持长宽比resize
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(nw, nh));
    
    // 添加padding
    int dw = (input_width - nw) / 2;
    int dh = (input_height - nh) / 2;
    cv::Mat padded = cv::Mat::zeros(input_height, input_width, CV_8UC3);
    padded.setTo(cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(dw, dh, nw, nh)));
    
    // 转换为模型输入格式
    cv::dnn::blobFromImage(padded, blob, 1/255.0, cv::Size(), cv::Scalar(0, 0, 0), true, true);
}

// 仿射变换矩阵
float i2d_[6]; // image to dst(network)
float d2i_[6]; // dst to image

void ObjectDetector::preprocessNanoDet(const cv::Mat& image, cv::Mat& blob, float& scale, int& nw, int& nh) {
    // 计算缩放比例
    float scale_x = (float)input_width / (float)image.cols;
    float scale_y = (float)input_height / (float)image.rows;
    scale = std::min(scale_x, scale_y);
    
    // 计算仿射变换矩阵
    i2d_[0] = scale;  i2d_[1] = 0;  i2d_[2] = (-scale * image.cols + input_width + scale - 1) * 0.5;
    i2d_[3] = 0;  i2d_[4] = scale;  i2d_[5] = (-scale * image.rows + input_height + scale - 1) * 0.5;
    
    // 计算逆变换矩阵
    cv::Mat m2x3_i2d(2, 3, CV_32F, i2d_);
    cv::Mat m2x3_d2i(2, 3, CV_32F, d2i_);
    cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
    
    // 执行仿射变换
    cv::Mat resized;
    cv::warpAffine(image, resized, m2x3_i2d, cv::Size(input_width, input_height),
                  cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(114));
    
    // 转换为float32并归一化
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32FC3);
    
    // 减去均值并除以标准差
    float mean[] = {103.53f, 116.28f, 123.675f};
    float std[] = {57.375f, 57.12f, 58.395f};
    for (int i = 0; i < resized.rows; ++i) {
        for (int j = 0; j < resized.cols; ++j) {
            cv::Vec3f& pixel = float_img.at<cv::Vec3f>(i, j);
            pixel[0] = (pixel[0] - mean[0]) / std[0];
            pixel[1] = (pixel[1] - mean[1]) / std[1];
            pixel[2] = (pixel[2] - mean[2]) / std[2];
        }
    }
    
    // 转换为CHW格式
    std::vector<cv::Mat> channels;
    cv::split(float_img, channels);
    cv::merge(channels, blob);
    cv::dnn::blobFromImage(blob, blob); // 添加batch维度
    
    nw = input_width;
    nh = input_height;
    scale = 1.0f;
}

// 快速指数计算
float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

// Softmax激活函数
void activation_function_softmax(const float *src, float *dst, int length) {
    const float alpha = *std::max_element(src, src + length);
    float denominator{0};
    for (int i = 0; i < length; ++i) {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }
    for (int i = 0; i < length; ++i) {
        dst[i] /= denominator;
    }
}

// 解码预测框
Box disPred2Bbox(const float* dfl_det, int label, float score, int x, int y, int stride, int input_width, int input_height) {
    float ct_x = x * stride;
    float ct_y = y * stride;
    std::vector<float> dis_pred;
    dis_pred.reserve(4);
    float dis_after_sm[16 + 1]; // reg_max + 1
    
    for (int i = 0; i < 4; i++) {
        float dis = 0;
        activation_function_softmax(dfl_det + i * (16 + 1), dis_after_sm, 16 + 1);
        for (int j = 0; j < 16 + 1; j++) {
            dis += j * dis_after_sm[j];
        }
        dis *= stride;
        dis_pred[i] = dis;
    }
    
    float xmin = std::max(ct_x - dis_pred[0], 0.f);
    float ymin = std::max(ct_y - dis_pred[1], 0.f);
    float xmax = std::min(ct_x + dis_pred[2], (float)input_width);
    float ymax = std::min(ct_y + dis_pred[3], (float)input_height);
    
    // 转换回原图坐标
    float image_x1 = d2i_[0] * xmin + d2i_[2];
    float image_y1 = d2i_[0] * ymin + d2i_[5];
    float image_x2 = d2i_[0] * xmax + d2i_[2];
    float image_y2 = d2i_[0] * ymax + d2i_[5];
    
    return Box{image_x1, image_y1, image_x2, image_y2, score, label};
}


// yolov8后处理
std::vector<DetectionResult> ObjectDetector::postprocessYOLOv8(const ov::Tensor& output_tensor, float scale, 
                                                              int nw, int nh, int image_width, int image_height) {
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


// nanodet后处理
std::vector<DetectionResult> ObjectDetector::postprocessNanoDet(const ov::Tensor& output_tensor, float /*scale*/,
                                                               int /*nw*/, int /*nh*/, int image_width, int image_height) {
    const float* output_data = output_tensor.data<const float>();
    int num_anchors = output_tensor.get_shape()[1]; // 3598
    int num_values = output_tensor.get_shape()[2];  // 33
    
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    
    // 生成网格中心点(类似示例代码中的generate_grid_center_priors)
    std::vector<CenterPrior> center_priors;
    for (int stride : {8, 16, 32}) {
        int feat_w = std::ceil((float)input_width / (float)stride);
        int feat_h = std::ceil((float)input_height / (float)stride);
        for (int y = 0; y < feat_h; y++) {
            for (int x = 0; x < feat_w; x++) {
                center_priors.push_back({x, y, stride});
            }
        }
    }
    
    for (int i = 0; i < num_anchors; ++i) {
        // 获取分类分数
        float score = output_data[i * num_values];
        
        if (score > 0.5f) {
            // 获取当前中心点信息
            int x = center_priors[i].x;
            int y = center_priors[i].y;
            int stride = center_priors[i].stride;
            
            // 解码预测框
            Box box = disPred2Bbox(output_data + i * num_values + num_class_,
                                 0, score, x, y, stride, input_width, input_height);
            
            // 跳过无效框
            if(box.label == -1) continue;
            
            // 转换为cv::Rect
            boxes.emplace_back(static_cast<int>(box.x1), 
                             static_cast<int>(box.y1),
                             static_cast<int>(box.x2 - box.x1),
                             static_cast<int>(box.y2 - box.y1));
            scores.push_back(score);
            class_ids.push_back(0); // 目前只检测排球一个类别
            
            // 调试输出
            std::cout << "Detected - x1:" << box.x1 << " y1:" << box.y1 
                     << " x2:" << box.x2 << " y2:" << box.y2 << std::endl;
        }
    }
    
    // 应用NMS过滤重复检测
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
