import cv2
import numpy as np
import time
from openvino import Core

# 初始化OpenVINO核心
ie = Core()

# 加载模型
model_xml = "./data/best_openvino_model/best.xml"
model_bin = "./data/best_openvino_model/best.bin"
model = ie.read_model(model=model_xml, weights=model_bin)
compiled_model = ie.compile_model(model=model, device_name="CPU")

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
    # 保持长宽比resize
    h, w = image.shape[:2]
    scale = min(W / w, H / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (nw, nh))
    
    # 添加padding
    dw, dh = (W - nw) // 2, (H - nh) // 2
    padded = np.full((H, W, 3), 114, dtype=np.uint8)
    padded[dh:dh+nh, dw:dw+nw] = resized
    
    # 转换为模型输入格式
    blob = cv2.dnn.blobFromImage(padded, 1/255.0, (W, H), swapRB=True)
    return blob, (scale, (dw, dh))

def postprocess(outputs, scale, padding):
    # 打印前5个anchor的5个值
    arr = np.squeeze(outputs).T
    print("First 5 anchors:")
    for i in range(5):
        print(f"Anchor {i}: ", arr[i, :5])
    
    # 解析输出
    boxes = []
    confidences = []
    class_ids = []
    
    # 这里需要根据实际模型输出结构调整
    # 假设输出是1x5x8400格式
    # 1 是批次大小，5是每个anchor的参数（cx, cy, w, h, confidence + class scores），8400是anchor数量
    # outputs (1, 5, 8400)   squeeze(outputs)降维后(5, 8400)     .T转置后 (8400, 5)

    outputs = np.squeeze(outputs).T
    for output in outputs:
        class_scores = output[4:]
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]
        
        if confidence > 0.5:  # 置信度阈值
            cx, cy, w, h = output[0], output[1], output[2], output[3]
            
            # 调整到原始图像尺寸
            dw, dh = padding
            x1 = int((cx - w / 2 - dw) / scale)
            y1 = int((cy - h / 2 - dh) / scale)
            x2 = int((cx + w / 2 - dw) / scale)
            y2 = int((cy + h / 2 - dh) / scale)
            
            boxes.append([x1, y1, x2, y2])
            class_ids.append(class_id)
            confidences.append(float(confidence))
    
    # 确保boxes和confidences长度一致
    if len(boxes) != len(confidences):
        return []
    
    # NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
    
    results = []
    for i in indices:
        box = boxes[i]
        if i < len(class_ids) and i < len(confidences):
            results.append({
                "class_id": class_ids[i],
                "confidence": confidences[i],
                "box": box
            })
    
    return results

# 主循环
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 预处理
    blob, (scale, padding) = preprocess(frame)
    
    # 推理
    start = time.time()
    outputs = compiled_model([blob])[output_layer]
    end = time.time()
    fps = 1 / (end - start)
    
    # 后处理
    detections = postprocess(outputs, scale, padding)
    
    # 绘制结果
    for det in detections:
        box = det["box"]
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), colors[0], 2)
        label = f"{class_names[det['class_id']]}: {det['confidence']:.2f}"
        cv2.putText(frame, label, (box[0], box[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[0], 2)
    
    # 显示FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("YOLOv8 OpenVINO Inference", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
