import sys
sys.path.append('./MvImport')
from MvCameraControl_class import MvCamera, MV_GIGE_DEVICE, MV_USB_DEVICE, MV_CC_DEVICE_INFO_LIST, MV_CC_DEVICE_INFO, MV_TRIGGER_MODE_OFF, MV_ACCESS_Exclusive, MV_FRAME_OUT
import ctypes
from ctypes import *
import time
import cv2
import numpy as np
# sys.path.append("./Become a master in a hundred days/CAMapi")
class CameraManager:
 
    def __init__(self):
        self.cam = None
        self.data_buf = None
        self.device_status = False
        self.stOutFrame = None
 
    def data_camera(self):
 
        # 枚举设备
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        deviceList = MV_CC_DEVICE_INFO_LIST()
        # 实例相机
        self.cam = MvCamera()
        ret = self.cam.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret is None:
            print("错误")
            return
        else:
            print(ret)
        # 选择设备
        stDeviceList = cast(deviceList.pDeviceInfo[int(0)], POINTER(MV_CC_DEVICE_INFO)).contents
        # 创建句柄
        ret = self.cam.MV_CC_CreateHandleWithoutLog(stDeviceList)
        # start_time = time.time()
        # 获取设备名称,返回是一个内存地址,循环遍历用chr把每个字符码转换为字符
        # strModeName = ""
        # for per in stDeviceList.SpecialInfo.stGigEInfo.chModelName:
        #     strModeName = strModeName + chr(per)
        # print(f"device model name:{strModeName}")
        # 获取设备名称,ctypes.string_at 函数直接将内存地址中的内容读取为字节字符串，然后使用 decode('utf-8') 进行解码。
        strModeName = ctypes.string_at(stDeviceList.SpecialInfo.stGigEInfo.chModelName).decode('utf-8')
        print(f"设备名称:{strModeName}")
        # end_time = time.time()
        # camera_time = round(abs(start_time - end_time) * 1000, 3)  # 保留小数点后3为,拍照时间
        # print(f"获取设备名称时间:{camera_time}ms")
        # 设置触发方式
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        # 打开相机
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        print("打开相机执行码:[0x%x]" % ret)
        # 开始取流
        ret = self.cam.MV_CC_StartGrabbing()
        self.device_status = True
        return self.cam
 
 
    def get_image(self):
 
        # 获取一张图像
        self.stOutFrame = MV_FRAME_OUT()  # 图像结构体,输出图像地址&信息
        start_time = time.time()
        ret = self.cam.MV_CC_GetImageBuffer(self.stOutFrame, 300)  # 图像获取函数
        print("图像获取执行码:[0x%x]" % ret)
        end_time = time.time()
        camera_time = round(abs(start_time - end_time) * 1000, 3)  # 保留小数点后3为,拍照时间
        print(f"获取图像时间:{camera_time}ms")

        nPayloadSize = self.stOutFrame.stFrameInfo.nFrameLen
        pData = self.stOutFrame.pBufAddr
        print(f"nPayloadSize: {nPayloadSize}, pData: {pData}")
        if not pData or nPayloadSize == 0:
            print("图像数据指针无效或长度为0！")
            self.cam.MV_CC_FreeImageBuffer(self.stOutFrame)
            return None

        self.data_buf = (c_ubyte * nPayloadSize)()
        try:
            src = ctypes.cast(pData, ctypes.POINTER(ctypes.c_ubyte * nPayloadSize))
            ctypes.memmove(self.data_buf, src.contents, nPayloadSize)
        except Exception as e:
            print(f"memmove异常: {e}")
            self.cam.MV_CC_FreeImageBuffer(self.stOutFrame)
            return None

        end_time = time.time()
        camera_time = round(abs(start_time - end_time) * 1000, 3)
        print(f"获取图像储存时间:{camera_time}ms")
        self.cam.MV_CC_FreeImageBuffer(self.stOutFrame)
        return self.data_buf
 
 
    def off_camera(self):
 
        # 停止取流
        ret = self.cam.MV_CC_StopGrabbing()
        print("停止取流执行码:[0x%x]" % ret)
 
        # 关闭设备
        ret = self.cam.MV_CC_CloseDevice()
        print("关闭设备执行码:[0x%x]" % ret)
 
        # 销毁句柄
        ret = self.cam.MV_CC_DestroyHandle()
        print("销毁句柄执行码:[0x%x]" % ret)
        self.device_status = False
        return self.device_status
    
 
 
 
#实例化类
CAM = CameraManager()
deta_CAM = input("输入1链接相机:")
if deta_CAM == "1":
    CAM.data_camera()
    print(f"当前相机链接状态:{CAM.device_status}")
else:
    print("链接相机错误!")
 
 
 
deta_CAM = input("输入2获取图片:")
if deta_CAM == "2":
    CAM.get_image()
    print(f"当前相机链接状态:{CAM.device_status}获取图片!")
else:
    print("获取图片错误!")
 
# 将 c_ubyte 数组转换为 numpy 数组
if CAM.data_buf is not None:
    temp = np.frombuffer(CAM.data_buf, dtype=np.uint8)
    width = CAM.stOutFrame.stFrameInfo.nWidth
    height = CAM.stOutFrame.stFrameInfo.nHeight
    print(width)
    print(height)
    # 自动判断通道数
    try:
        if temp.size == width * height:
            temp = temp.reshape((height, width))
            print("检测到单通道灰度图像")
            cv2.namedWindow("ori", cv2.WINDOW_NORMAL)
            cv2.imshow("ori", temp)
        elif temp.size == width * height * 3:
            temp = temp.reshape((height, width, 3))
            print("检测到三通道彩色图像")
            # 将 BGR 转换为 RGB
            temp = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)
            cv2.namedWindow("ori", cv2.WINDOW_NORMAL)
            cv2.imshow("ori", temp)
        else:
            print(f"数据长度与分辨率不符，size={temp.size}, 期望={width*height}或{width*height*3}")
    except Exception as e:
        print(f"Reshape error: {e}")
    # 灰度图像可选显示
    if len(temp.shape) == 2:
        gray = temp
        cv2.namedWindow("gray", cv2.WINDOW_NORMAL)
        cv2.imshow("gray", gray)
    elif len(temp.shape) == 3:
        gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        cv2.namedWindow("gray", cv2.WINDOW_NORMAL)
        cv2.imshow("gray", gray)
    cv2.waitKey(0)
else:
    print("未获取到有效图像数据，无法显示！")
 
 
 
deta_CAM = input("输入3关闭相机设备:")
if deta_CAM == "3":
    CAM.off_camera()
    print(f"当前相机链接状态:{CAM.device_status}")
else:
    print("关闭相机错误!")