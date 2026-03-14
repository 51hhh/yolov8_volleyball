cmake_minimum_required(VERSION 3.10)
project(volleyball_detector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# 查找依赖包
set(BOOST_ROOT /opt/boost_1_74_0)
find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)

# 添加源文件
set(SOURCES
    camera/hikvision_wrapper.cpp
    detect/detect.cpp
    serial/serial.cpp
    main.cpp
)

# 添加可执行文件
add_executable(volleyball_detector
    ${SOURCES}
)

# 包含目录
target_include_directories(volleyball_detector PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/camera
    ${CMAKE_CURRENT_SOURCE_DIR}/detect
    ${CMAKE_CURRENT_SOURCE_DIR}/general
    ${CMAKE_CURRENT_SOURCE_DIR}/serial
    ${OpenVINO_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    /opt/MVS/include/
)

# 添加编译定义
target_compile_definitions(volleyball_detector PRIVATE
    USE_OPENVINO
    USE_OPENCV
)

# 添加海康SDK库路径
link_directories(/opt/MVS/bin)

# 链接库
target_link_libraries(volleyball_detector PRIVATE
    openvino::runtime
    ${OpenCV_LIBS}
    Boost::system
    MvCameraControl
    ${OpenVINO_LIBRARIES}
)

# 安装规则
install(TARGETS volleyball_detector DESTINATION bin)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/config.json DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
