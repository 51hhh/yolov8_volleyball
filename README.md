# YOLOv8 Volleyball 排球检测系统

基于 **YOLOv8 / NanoDet** 与 **Intel OpenVINO** 推理引擎的排球实时检测系统，通过 **串口通信** 将检测坐标发送至下位机控制系统，实现视觉引导的自动追球。支持多模型切换、多摄像头类型、JSON 配置驱动，一套代码适配多种部署场景。

## 特性

- **多检测器支持**：YOLOv8 与 NanoDet 双引擎，统一 ObjectDetector 接口
- **多精度模型**：FP32 / INT8 量化 / ShuffleNetV2 轻量骨干网络
- **JSON 配置驱动**：模型路径、检测器类型、摄像头类型、调试模式均通过 `config.json` 配置
- **海康工业相机 + USB 摄像头**：PIMPL 封装海康 SDK，自动切换摄像头类型
- **串口通信**：Boost.Asio 异步串口，支持二进制和 ASCII 协议，输出球心偏移坐标
- **Python 推理版本**：AsyncInferQueue 异步推理，支持独立部署
- **DEBUG 模式**：可视化检测框、置信度、FPS 显示

## 系统集成

本项目作为上位机视觉系统，与 [RM-C-board-volleyball](https://github.com/51hhh/RM-C-board-volleyball) 下位机协同工作：

```
┌─────────────────────────┐                    ┌─────────────────────────┐
│  YOLOv8 Volleyball      │   UART 115200bps   │  RM-C-board-volleyball  │
│  (上位机 - x86/ARM)     │ ──────────────────▶ │  (下位机 - STM32F407)   │
│                         │   "x,y\n"           │                         │
│  相机 → 检测 → 串口输出  │                    │  解析坐标 → 底盘运动    │
│                         │                    │  → 麦轮全向控制          │
└─────────────────────────┘                    └─────────────────────────┘
```

## 项目结构

```
yolov8_volleyball/
├── main.cpp                       # 主程序：配置加载 / 相机初始化 / 检测循环 / 串口输出
├── detect/
│   ├── detect.hpp                 # ObjectDetector 类定义（YOLOv8 + NanoDet）
│   └── detect.cpp                 # 检测实现：预处理 / 推理 / 后处理
├── camera/
│   ├── hikvision_wrapper.hpp      # 海康相机 PIMPL 接口
│   └── hikvision_wrapper.cpp      # 海康相机实现
├── serial/
│   ├── serial.hpp                 # Boost.Asio 串口类
│   └── serial.cpp                 # 串口初始化 / 二进制发送 / ASCII 发送
├── general/
│   ├── common_struct.hpp          # 公共数据结构
│   ├── structs.hpp                # 串口相关结构定义
│   ├── debug.hpp                  # 调试宏与默认配置
│   └── nlohmann/json.hpp          # JSON 解析库
├── data/
│   ├── best_openvino_model/       # YOLOv8 FP32 模型
│   ├── best_int8_openvino_model/  # YOLOv8 INT8 量化模型
│   ├── best_openvino_model_shufv2/# ShuffleNetV2 骨干模型
│   └── best_openvino_nanodet/     # NanoDet IR 模型
├── config.json                    # 运行时配置文件
├── yolov8_openvino_inference.py   # Python 异步推理版本
├── start.sh                       # 一键启动脚本
├── MvImport/                      # 海康 Python SDK
└── CMakeLists.txt                 # CMake 构建配置
```

## 配置文件

`config.json` 控制所有运行时行为：

```json
{
    "path": {
        "xml_file_path": "data/best_openvino_model/best.xml",
        "bin_file_path": "data/best_openvino_model/best.bin",
        "int8_xml_file_path": "data/best_int8_openvino_model/best.xml",
        "int8_bin_file_path": "data/best_int8_openvino_model/best.bin",
        "nanodet_xml_path": "data/best_openvino_nanodet/nanodet_ir.xml",
        "nanodet_bin_path": "data/best_openvino_nanodet/nanodet_ir.bin",
        "serial_path": "/dev/ttyUSB0"
    },
    "model_type": "fp32",
    "detector_type": "yolov8",
    "camera_type": "hikvision",
    "DEBUG": false
}
```

| 字段 | 可选值 | 说明 |
|------|--------|------|
| `model_type` | `fp32` / `int8` | 模型精度（仅 YOLOv8 生效） |
| `detector_type` | `yolov8` / `nanodet` | 检测器类型 |
| `camera_type` | `hikvision` / `usb` | 摄像头类型 |
| `DEBUG` | `true` / `false` | 是否显示可视化窗口 |

## 环境要求

| 依赖 | 说明 |
|------|------|
| Intel OpenVINO | >= 2022.1 |
| OpenCV | >= 4.0 |
| Boost | >= 1.74（`system` 组件，用于串口） |
| 海康 MVS SDK | `/opt/MVS/`（海康相机模式） |
| CMake | >= 3.10 |
| GCC | >= 7.0（C++17） |

## 快速开始

### 1. 编译

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### 2. 配置

编辑 `config.json` 选择模型和摄像头类型。

### 3. 运行

**C++ 版本：**
```bash
./start.sh
# 或直接运行
./volleyball_detector
```

**Python 异步推理版本：**
```bash
python yolov8_openvino_inference.py
```

## 预训练模型

| 模型 | 目录 | 特点 |
|------|------|------|
| YOLOv8 FP32 | `data/best_openvino_model/` | 最高精度 |
| YOLOv8 INT8 | `data/best_int8_openvino_model/` | 量化加速，适合边缘部署 |
| ShuffleNetV2 | `data/best_openvino_model_shufv2/` | 轻量骨干，极致速度 |
| NanoDet | `data/best_openvino_nanodet/` | 超轻量级，低算力平台 |

所有模型均针对排球单类别检测训练。

## 串口通信协议

### 输出格式（上位机 → 下位机）

```
格式:  "{delta_x},{delta_y}\n"
示例:  "120,-50\n"
说明:  排球中心相对画面中心的偏移量
       delta_x > 0 表示目标在右侧
       delta_y > 0 表示目标在上方
无目标: "0,0\n"
```

### 坐标计算

```cpp
out_x = box_center_x - image_center_x   // 右为正
out_y = image_center_y - box_center_y   // 上为正
```

## 技术细节

### 多检测器统一接口

`ObjectDetector` 类通过 `ModelType` 枚举统一 YOLOv8 和 NanoDet 的预处理与后处理逻辑，切换检测器只需修改 `config.json`，无需重新编译。

### Python 异步推理

`yolov8_openvino_inference.py` 使用 OpenVINO `AsyncInferQueue` 实现异步推理流水线，在等待推理结果的同时进行下一帧的预处理，提升吞吐量。

## 许可证

Apache License 2.0
