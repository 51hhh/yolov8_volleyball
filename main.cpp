#include <iostream>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <fstream>
#include "detect.hpp"
#include "camera.hpp"
#include "nlohmann/json.hpp"
#include "serial.hpp"
#include "opencv2/opencv.hpp"

#define PROJECT_PATH "/home/rick/yolo_new/"
#define DEBUG 1

std::atomic<bool> state(true);      // 原子变量，用于通知线程终止
toe::hik_camera hik_cam;            // 创建海康相机的对象
toe::serial serial;                 // 创建串口的对象
toe::ov_detect ov_detector;         // 创建目标检测器的对象
nlohmann::json config;              // 配置文件

// 监控命令行ctrl+c,用于手动退出
void sigint_handler(int sig)
{
    if (sig == SIGINT)
    {
        state.store(false);
    }
}
// 串口线程（预留接口）
void serial_process() {
    while (state.load()) {
        // 预留串口处理逻辑
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

// 检测处理线程
void detect_process() {
    cv::Mat frame;
    int color = 0;
    ov_detector.detect_init(config, color);

    while (state.load()) {
        // 获取图像并执行检测
        mutex_array[0].lock();
        frame = frame_array[0];
        mutex_array[0].unlock();

        if (!frame.empty() && frame.rows > 0 && frame.cols > 0) {
            try {
                ov_detector.input_img = frame.clone();  // 确保使用深拷贝
                ov_detector.detect();
            } catch (const std::exception& e) {
                std::cerr << "Detection error: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "Unknown detection error" << std::endl;
            }
        }
    }
}

// 图像采集线程
void grab_img() {
    if(!hik_cam.hik_init(config, 0)) {
        std::cerr << "Failed to initialize camera" << std::endl;
        return;
    }

    while (state.load()) {
        // 图像采集逻辑
        mutex_array[0].lock();
        cv::Mat frame = frame_array[0];
        mutex_array[0].unlock();
        
        if (!frame.empty()) {
            ov_detector.push_img(frame);
        }
    }
    hik_cam.hik_end();
}

int main() {
    // 初始化全局变量
    state.store(true);
    // 为了提升cout的效率关掉缓存区同步，此时就不能使用c风格的输入输出了，例如printf
    // oi上常用的技巧，还有提升输出效率的就是减少std::endl和std::flush的使用
    std::ios::sync_with_stdio(false);
    std::cin.tie(0);

    // 加载配置
    std::ifstream f(std::string(PROJECT_PATH) + "config.json");
    config = nlohmann::json::parse(f);

    // 启动线程
    std::thread grab_thread(grab_img);
    std::thread detect_thread(detect_process);
    std::thread serial_thread(serial_process);

    // 主线程处理显示
    cv::Mat display_frame;
    while (state.load()) {
        mutex_array[0].lock();
        if(!frame_array[0].empty()) {
            display_frame = frame_array[0].clone();
        }
        mutex_array[0].unlock();
        
        if(!display_frame.empty()) {
            cv::imshow("Detection", display_frame);
            if (cv::waitKey(1) == 27) break;
        }
        
        // 检查线程状态
        if (!grab_thread.joinable() || !detect_thread.joinable()) {
            state.store(false);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // 等待线程结束
    grab_thread.join();
    detect_thread.join();
    serial_thread.join();

    return 0;
}
