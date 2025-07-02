import cv2
import numpy as np
import time
from openvino import Core

# 初始化OpenVINO核心
ie = Core()

# 设置性能参数
ie.set_property("CPU", {"PERFORMANCE_HINT": "THROUGHPUT"})
ie.set_property("CPU", {"NUM_STREAMS": "4"})

# 加载模型
model_xml = "./data/best_openvino_model/best.xml"
model_bin = "./data/best_openvino_model/best.bin"
model = ie.read_model(model=model_xml, weights=model_bin)
compiled_model = ie.compile_model(model=model, device_name="CPU")

# 准备异步推理
from openvino import AsyncInferQueue
infer_queue = AsyncInferQueue(compiled_model, jobs=4)

# 获取输入输出节点
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)
input_shape = input_layer.shape
H, W = input_shape[2], input_shape[3]

# 初始化摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 类别名称
class_names = ["volleyball"]

# 颜色设置
colors = [(0, 0, 255)]

def preprocess(image):
    try:
        # 使用UMat加速
        image_umat = cv2.UMat(image)
        
        # 保持长宽比resize
        h, w = image.shape[:2]
        scale = min(W / w, H / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(image_umat, (nw, nh))
        
        # 使用向量化操作添加padding
        dw, dh = (W - nw) // 2, (H - nh) // 2
        
        # 修正UMat初始化方式
        padded = cv2.UMat(H, W, cv2.CV_8UC3)
        padded.setTo(114)
        padded_roi = padded[dh:dh+nh, dw:dw+nw]
        resized.copyTo(padded_roi)
        
        # 转换为模型输入格式
        padded_mat = padded.get()
        blob = cv2.dnn.blobFromImage(padded_mat, 1/255.0, (W, H), swapRB=True)
        return blob, (scale, (dw, dh))
        
    except Exception as e:
        print(f"UMat processing failed, falling back to Mat: {e}")
        # 回退到常规Mat处理
        resized = cv2.resize(image, (nw, nh))
        padded = np.full((H, W, 3), 114, dtype=np.uint8)
        padded[dh:dh+nh, dw:dw+nw] = resized
        blob = cv2.dnn.blobFromImage(padded, 1/255.0, (W, H), swapRB=True)
        return blob, (scale, (dw, dh))

def postprocess(outputs, scale, padding):
    # 向量化处理输出
    outputs = np.squeeze(outputs).T
    dw, dh = padding
    
    # 提取所有预测框
    predictions = outputs[outputs[:, 4:].max(axis=1) > 0.5]  # 置信度阈值0.5
    if len(predictions) == 0:
        return []
    
    # 获取类别和置信度
    class_scores = predictions[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(len(class_ids)), class_ids]
    
    # 计算边界框坐标 (向量化操作)
    cx, cy, w, h = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]
    x1 = ((cx - w / 2 - dw) / scale).astype(int)
    y1 = ((cy - h / 2 - dh) / scale).astype(int)
    x2 = ((cx + w / 2 - dw) / scale).astype(int)
    y2 = ((cy + h / 2 - dh) / scale).astype(int)
    boxes = np.column_stack((x1, y1, x2, y2))
    
    # NMS处理
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), 0.5, 0.5)
    
    # 构建结果
    results = []
    for i in indices:
        idx = i if isinstance(i, np.int32) else i[0]
        results.append({
            "class_id": class_ids[idx],
            "confidence": confidences[idx],
            "box": boxes[idx].tolist()
        })
    
    return results

# 帧缓冲和结果处理
MAX_BUFFER_SIZE = 5  # 最大缓冲帧数
frame_buffer = {}
latest_result = None
fps_counter = 0
fps = 0
last_time = time.time()
total_frames = 0
processing_times = []

def completion_callback(infer_request, frame_id):
    global latest_result, fps_counter, processing_times
    start_time = time.time()
    
    outputs = infer_request.get_output_tensor(0).data
    if frame_id not in frame_buffer:
        return
        
    blob, (scale, padding), frame = frame_buffer[frame_id]
    detections = postprocess(outputs, scale, padding)
    latest_result = (frame, detections)  # (原始帧, 检测结果)
    
    fps_counter += 1
    processing_time = time.time() - start_time
    processing_times.append(processing_time)
    del frame_buffer[frame_id]

def print_stats():
    if len(processing_times) > 0:
        avg_time = sum(processing_times) / len(processing_times)
        print(f"\nPerformance Stats:")
        print(f"Average processing time: {avg_time*1000:.2f}ms")
        print(f"Max FPS achieved: {fps:.2f}")
        print(f"Total frames processed: {total_frames}")

infer_queue.set_callback(completion_callback)

# 主循环
frame_id = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        total_frames += 1
        
        # 如果缓冲满了，丢弃最旧的帧
        if len(frame_buffer) >= MAX_BUFFER_SIZE:
            oldest_id = min(frame_buffer.keys())
            del frame_buffer[oldest_id]
            print(f"Warning: Frame buffer full, dropped frame {oldest_id}")
        
        # 预处理
        blob, (scale, padding) = preprocess(frame)
        frame_buffer[frame_id] = (blob, (scale, padding), frame.copy())
        
        # 提交异步推理
        infer_queue.start_async({input_layer: blob}, frame_id)
        frame_id += 1
    
        # 处理已完成的结果
        current_time = time.time()
        if current_time - last_time >= 1.0:
            fps = fps_counter / (current_time - last_time)
            fps_counter = 0
            last_time = current_time
        
        if latest_result:
            result_frame, detections = latest_result
            
            # 绘制结果
            for det in detections:
                box = det["box"]
                cv2.rectangle(result_frame, (box[0], box[1]), (box[2], box[3]), colors[0], 2)
                label = f"{class_names[det['class_id']]}: {det['confidence']:.2f}"
                cv2.putText(result_frame, label, (box[0], box[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[0], 2)
            
            # 显示FPS
            cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("YOLOv8 OpenVINO Inference", result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nInterrupted by user")
finally:
    # 确保所有资源正确释放
    cap.release()
    cv2.destroyAllWindows()
    print_stats()
