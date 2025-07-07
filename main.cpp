#include "detect/detect.hpp"
#include "camera/hikvision_wrapper.hpp"
#include "general/common_struct.hpp"
#include "serial/serial.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>

int main() {
    // 读取配置文件
    std::ifstream config_file("config.json");
    nlohmann::json config;
    config_file >> config;
    std::string camera_type = config.value("camera_type", "usb");
    std::string model_type = config["model_type"];
    std::string xml_path, bin_path;
    if (model_type == "int8") {
        xml_path = config["path"]["int8_xml_file_path"];
        bin_path = config["path"]["int8_bin_file_path"];
    } else {
        xml_path = config["path"]["xml_file_path"];
        bin_path = config["path"]["bin_file_path"];
    }

    // 初始化检测器
    YOLOv8Detector detector(xml_path, bin_path);


    //  frame矩阵
    cv::Mat frame;
    //  VideoCapture对象
    cv::VideoCapture cap;
    //  海康摄像头对象
    HikVisionWrapper* hik = nullptr;
    //  上一帧的时间
    auto prev_time = std::chrono::high_resolution_clock::now();



    if (camera_type == "hikvision") {
        // 调用海康摄像头
        int cam_id = 0;
        if (config.contains("camera") && config["camera"].contains("0")) {
            auto c = config["camera"]["0"];
        }
        s_camera_params params{cam_id, 640, 480, 0, 0, 10000};
        hik = new HikVisionWrapper(params);
        if (!hik->initialize()) {
            fprintf(stderr, "Failed to initialize Hikvision camera %d\n", cam_id);
            return -1;
        }
    } else {
        // 调用USB摄像头
        cap.open(0);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open camera" << std::endl;
            return -1;
        }
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    }

    // 串口对象初始化
    toe::serial serial;
    serial.init_port(config);

    // 主循环
    while (true) {
        if (camera_type == "hikvision"){
            //使用海康获取帧
            if (!hik->getFrame(frame)) {
                fprintf(stderr, "Hikvision camera get frame failed.\n");
                break;
            }
        }
        else{
            //使用USB摄像头获取帧
            if (!cap.read(frame)) {
                std::cerr << "Error: Could not read frame" << std::endl;
                break;
            }
        }

        // 计算FPS
        auto curr_time = std::chrono::high_resolution_clock::now();
        float fps = 1e6 / std::chrono::duration_cast<std::chrono::microseconds>(curr_time - prev_time).count();
        prev_time = curr_time;

        // 执行检测
        auto detections = detector.detect(frame);

        // 串口输出评分最高的目标中心坐标
        if (!detections.empty()) {
            auto max_it = std::max_element(detections.begin(), detections.end(),
                [](const auto& a, const auto& b) { return a.confidence < b.confidence; });
            const auto& det = *max_it;
            int img_cx = frame.cols / 2;
            int img_cy = frame.rows / 2;
            int box_cx = det.box.x + det.box.width / 2;
            int box_cy = det.box.y + det.box.height / 2;
            int out_x = box_cx - img_cx; // 右为正，左为负
            int out_y = img_cy - box_cy; // 上为正，下为负


            // std::vector<double> msg = {static_cast<double>(out_x), static_cast<double>(out_y), 0.0};
            // serial.send_msg(msg);


            char buf[64];
            snprintf(buf, sizeof(buf), "%d,%d\n", out_x, out_y);
            serial.send_msg(buf, strlen(buf)); // 新增：以16进制ASCII方式发送字符串

        }
        else {
            // 如果没有检测到目标，发送空消息
            serial.send_msg("0,0\n", 4); // 新增：发送空消息
        }

        // 显示结果（仅在DEBUG模式）
        if (config["DEBUG"]) {

            // 绘制结果
            for (const auto& det : detections) {
                cv::rectangle(frame, det.box, cv::Scalar(0, 0, 255), 2);
                std::string label = "volleyball: " + std::to_string(det.confidence).substr(0, 4);
                cv::putText(frame, label, cv::Point(det.box.x, det.box.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
            }

            // 显示FPS
            cv::putText(frame, "FPS: " + std::to_string(fps).substr(0, 4), 
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
                    
            cv::imshow("YOLOv8 OpenVINO Inference", frame);
            // 退出条件
            if (cv::waitKey(1) == 'q') {
                break;
            }
        }
    }

    if (camera_type == "hikvision") {
        if (hik) {
            hik->release();
            delete hik;
        }
    } else {
        cap.release();
    }
    if (config["DEBUG"]) {
        cv::destroyAllWindows();
    }
    return 0;
}
