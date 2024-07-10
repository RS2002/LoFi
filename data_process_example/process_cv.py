import os
import tqdm
import cv2
import numpy as np
import pickle

# root="./cv"
root="/home/chentingwei/LoFi/lofi"


# 加载 YOLO 模型
net = cv2.dnn.readNet("./model/yolov3.weights", "./model/yolov3.cfg")
# 获取输出层的名称
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

src_points = np.array([[0, 0], [180, 0], [0, 480], [180, 480]], dtype="float32") # real world
dst_points = np.array([[222, 210], [374, 209], [65, 458], [495, 451]], dtype="float32") # image world


M = cv2.getPerspectiveTransform(src_points, dst_points)

data=[]

def get_gt(img_path,net):
    image = cv2.imread(img_path)
    # 加载图像
    height, width, channels = image.shape

    # 准备输入图像
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # 解析 YOLO 输出，找到人体边界框
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # 置信度阈值
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 如果检测到了多个框，只保留置信度最高的那个框
    if len(boxes) > 0:
        max_confidence_idx = np.argmax(confidences)
        boxes = [boxes[max_confidence_idx]]
    x, y, w, h = boxes[0]
    foot_position_image = (x + w // 2, y + h)

    person_img_coords = np.array([[foot_position_image[0], foot_position_image[1]]],
                                 dtype="float32")
    actual_coords = cv2.perspectiveTransform(np.array([person_img_coords]), np.linalg.inv(M))
    return actual_coords[0,0,0],actual_coords[0,0,1]


people_id=0
for people in os.listdir(root):
    print(people)
    path=os.path.join(root,people)

    pbar = tqdm.tqdm(os.listdir(path))

    x_list = []
    y_list = []
    img_path_list = []
    time_list = []

    for pic in pbar:
        if "color" not in pic:
            continue

        # print(pic)
        timestamp = pic.split("_")
        timestamp = timestamp[-1].split(".")
        timestamp = timestamp[0]
        timestamp = timestamp.split("-")
        # print(timestamp)
        timestamp = float(timestamp[0]) * 60 * 60 * 100 + float(timestamp[1]) * 60 * 100 + float(timestamp[2]) * 100 + float(timestamp[3])

        img_path = os.path.join(path, pic)
        x, y = get_gt(img_path, net)
        x_list.append(x)
        y_list.append(y)
        img_path_list.append(img_path)
        time_list.append(timestamp)

    data.append({
        'timestamp': np.array(time_list),
        'people_name': people,
        'people': people_id,
        'x': np.array(x_list),
        'y': np.array(y_list),
        'img_path': img_path_list
    })
    people_id += 1

output_file = './gt_data.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(data, f)
