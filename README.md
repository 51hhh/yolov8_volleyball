# YOLOv8 Volleyball 排球检测系统

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0)
[![Language](https://img.shields.io/badge/Language-C%2B%2B17-brightgreen.svg)]()
[![Platform](https://img.shields.io/badge/Platform-x86%20%7C%20ARM-lightgrey.svg)]()

基于 **YOLOv8 / NanoDet** 与 **Intel OpenVINO** 推理引擎的排球实时检测系统，通过 **Boost.Asio 串口通信** 将检测坐标发送至下位机控制系统，实现视觉引导的自动追球。支持多模型切换、多摄像头类型、JSON 配置驱动，是排球机器人系统中**功能最完整的上位机方案**。

> **项目定位**：推荐的上位机方案。提供真正的串口通信、JSON 配置文件、多检测器切换。轻量替代方案参见 [nanodet_volleyball](https://github.com/51hhh/nanodet_volleyball)，纯视觉调试参见 [Nanodet_OpenVINO](https://github.com/51hhh/Nanodet_OpenVINO)。

## 排球机器人系统全景

```
┌─────────────────────────┐                    ┌─────────────────────────┐
│  YOLOv8 Volleyball      │   UART 115200bps   │  RM-C-board-volleyball  │
│  (上位机 - 本项目)       │ ──────────────────▶ │  (下位机 - STM32F407)   │
│                         │   "out_x,out_y\n"   │                         │
│  相机 → 检测 → 串口输出  │                    │  解析坐标 → 底盘运动    │
│                         │                    │  → 麦轮全向控制          │
└─────────────────────────┘                    └─────────────────────────┘
```

| 仓库 | 角色 | 说明 |
|------|------|------|
| **yolov8_volleyball**（本项目） | 上位机（推荐） | YOLOv8+NanoDet 双引擎，Boost.Asio 串口通信 |
| [nanodet_volleyball](https://github.com/51hhh/nanodet_volleyball) | 上位机 | NanoDet 专用版，终端坐标输出 |
| [Nanodet_OpenVINO](https://github.com/51hhh/Nanodet_OpenVINO) | 上位机（调试用） | NanoDet 可视化调试版本 |
| [RM-C-board-volleyball](https://github.com/51hhh/RM-C-board-volleyball) | 下位机 | STM32F407 麦克纳姆轮底盘控制 |

## 特性

- **多检测器支持**：YOLOv8 与 NanoDet 双引擎，统一 ObjectDetector 接口
- **多精度模型**：FP32 / INT8 量化模型
- **JSON 配置驱动**：模型路径、检测器类型、摄像头类型、调试模式均通过 `config.json` 配置
- **海康工业相机 + USB 摄像头**：PIMPL 封装海康 SDK，通过配置切换
- **Boost.Asio 串口通信**：支持 ASCII 协议输出球心偏移坐标（二进制协议已实现但当前未启用）
- **Python 推理版本**：AsyncInferQueue 异步推理，支持独立部署
- **DEBUG 模式**：可视化检测框、置信度、FPS 显示，按 'q' 退出

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
│   └── best_openvino_nanodet/     # NanoDet IR 模型
├── config.json                    # 运行时配置文件
├── yolov8_openvino_inference.py   # Python 异步推理版本
├── start.sh                       # 启动脚本
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
| `model_type` | `fp32` / `int8` | 模型精度（仅 `detector_type=yolov8` 时生效） |
| `detector_type` | `yolov8` / `nanodet` | 检测器类型，切换无需重新编译 |
| `camera_type` | `hikvision` / `usb` | 摄像头类型 |
| `DEBUG` | `true` / `false` | 是否显示可视化窗口（开启后显示检测框、FPS） |

> **没有海康相机？** 将 `camera_type` 改为 `"usb"` 即可使用普通 USB 摄像头。
>
> **没有串口设备？** 将 `DEBUG` 改为 `true` 可单独测试检测效果。注意：当前版本在无串口设备时，串口初始化会不断重试，建议在代码中临时注释串口部分或确保 `/dev/ttyUSB0` 存在。

## 环境要求

| 依赖 | 版本要求 | 安装方式 |
|------|---------|---------|
| Intel OpenVINO | >= 2022.1 | [官方安装指南](https://docs.openvino.ai/latest/openvino_docs_install_guides_overview.html) |
| OpenCV | >= 4.0 | `sudo apt install libopencv-dev` |
| Boost | >= 1.74 | `sudo apt install libboost-system-dev` |
| 海康 MVS SDK | 最新版 | [海康机器人官网](https://www.hikrobotics.com/cn/machinevision/service/download)，安装到 `/opt/MVS/`（`camera_type=hikvision` 时需要） |
| CMake | >= 3.10 | `sudo apt install cmake` |
| GCC | >= 7.0 | C++17 支持 |

> 注：CMakeLists.txt 中 Boost 路径可能硬编码为 `/opt/boost_1_74_0`，如系统安装 Boost 路径不同，需修改 `CMakeLists.txt` 中的 `BOOST_ROOT`。

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/51hhh/yolov8_volleyball.git
cd yolov8_volleyball
```

### 2. 环境准备

```bash
# 设置 OpenVINO 环境变量（每次编译/运行前需要，或写入 ~/.bashrc）
source /opt/intel/openvino/setupvars.sh
```

### 3. 编译

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

> 注：编译产物名称为 `volleyball_detector`。`start.sh` 中引用的是旧版名称 `volleyball`，如需使用启动脚本请更新其中的可执行文件名。

### 4. 配置

编辑 `config.json` 选择模型和摄像头类型。首次测试建议：
```json
{
    "camera_type": "usb",
    "DEBUG": true
}
```

### 5. 运行

**C++ 版本：**
```bash
# 从 build 目录运行
./volleyball_detector

# 或从项目根目录
./build/volleyball_detector
```

**Python 异步推理版本（不需要串口设备）：**
```bash
python yolov8_openvino_inference.py
```

## 预训练模型

| 模型 | 目录 | config.json 配置 | 特点 |
|------|------|-----------------|------|
| YOLOv8 FP32 | `data/best_openvino_model/` | `detector_type=yolov8, model_type=fp32` | 最高精度 |
| YOLOv8 INT8 | `data/best_int8_openvino_model/` | `detector_type=yolov8, model_type=int8` | 量化加速 |
| NanoDet | `data/best_openvino_nanodet/` | `detector_type=nanodet` | 超轻量级 |

所有模型均针对排球单类别检测训练。

## 串口通信协议

### 输出格式（上位机 → 下位机）

```
格式:  "{out_x},{out_y}\n"
示例:  "120,-50\n"
说明:  排球中心相对画面中心的偏移量
       out_x > 0 表示目标在右侧
       out_y > 0 表示目标在上方
无目标: "0,0\n"
波特率: 115200 8N1
```

### 坐标计算

```cpp
out_x = box_center_x - image_center_x   // 右为正
out_y = image_center_y - box_center_y   // 上为正
```

## 技术细节

### 多检测器统一接口

`ObjectDetector` 类通过 `ModelType` 枚举统一 YOLOv8 和 NanoDet 的预处理与后处理逻辑，切换检测器只需修改 `config.json`，无需重新编译。

> **注意**：本项目中 NanoDet 后处理使用 3 尺度 (stride 8/16/32) 和 reg_max=16（17-bin Softmax），与独立 [Nanodet_OpenVINO](https://github.com/51hhh/Nanodet_OpenVINO) 项目的 4 尺度 (8/16/32/64) 和 reg_max=7（8-bin Softmax）不同。请确保使用与本项目匹配的 NanoDet 模型。

### Python 异步推理

`yolov8_openvino_inference.py` 使用 OpenVINO `AsyncInferQueue` 实现异步推理流水线，在等待推理结果的同时进行下一帧的预处理，提升吞吐量。

## 常见问题

| 问题 | 解决方案 |
|------|---------|
| 编译报错找不到 OpenVINO | `source /opt/intel/openvino/setupvars.sh` |
| 编译报错找不到 Boost | 检查 CMakeLists.txt 中 `BOOST_ROOT` 路径。如遇链接错误，确认 `target_link_libraries` 中包含 `Boost::system` |
| 串口打不开 / 程序卡住 | 检查 `/dev/ttyUSB0` 是否存在，`sudo chmod 666 /dev/ttyUSB0` |
| 没有串口设备 | 设置 `DEBUG=true` 单独测试检测效果，或注释 `main.cpp` 中串口相关代码 |
| 没有海康相机 | 将 `config.json` 中 `camera_type` 改为 `"usb"` |
| 检测不到排球 | 检查模型路径是否正确，降低代码中的置信度阈值 |
| start.sh 报错找不到 volleyball | 修改 `start.sh` 中的可执行文件名为 `volleyball_detector` |

## 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv8 模型
- [NanoDet](https://github.com/RangiLyu/nanodet) — 轻量级检测模型
- [Intel OpenVINO](https://github.com/openvinotoolkit/openvino) — 推理引擎
- [海康机器人](https://www.hikrobotics.com/) — 工业相机 SDK

## 许可证

本项目基于 [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0) 开源。
