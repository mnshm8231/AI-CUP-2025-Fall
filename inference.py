import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO

# 這裡model選擇可以直接best.pt，但highest private score 採取的策略為 選擇 "mAP50一定水準之下，recall較高的epoch"
# 該程式一次只跑一次推論，產生多個fold的預測需持續更改 MODEL_PATH 及 OUT_TXT

MODEL_PATH = "runs/detect_kfold_roi/fold1_roi/weights/epoch32.pt"
TEST_ROOT = "42_testing_image/testing_image"
OUT_TXT = "ensmble_roi_1.txt"

# ROI (from 100,153 → 317,382)
ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 100, 153, 317, 382
ROI_W = ROI_X2 - ROI_X1
ROI_H = ROI_Y2 - ROI_Y1


IMG_SIZE = 512  

model = YOLO(MODEL_PATH)
print("Model loaded.")


def infer_single_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠ 無法讀取: {img_path}")
        return []

    H, W, _ = img.shape

    # Crop ROI
    crop = img[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]

    # Resize
    crop_resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

    # YOLO 推論
    results = model.predict(
        crop_resized,
        batch=1,
        imgsz=IMG_SIZE,
        conf=0.001,
        iou=0.5,
        max_det=100,     
        verbose=False,
    )

    det = results[0].boxes
    if det is None or len(det) == 0:
        return []

    boxes = det.xyxy.cpu().numpy()
    confs = det.conf.cpu().numpy()

    outputs = []

    # 全部框映回原圖
    sx = ROI_W / IMG_SIZE
    sy = ROI_H / IMG_SIZE

    for (x1, y1, x2, y2), conf in zip(boxes, confs):
        X1 = x1 * sx + ROI_X1
        Y1 = y1 * sy + ROI_Y1
        X2 = x2 * sx + ROI_X1
        Y2 = y2 * sy + ROI_Y1

        # clamp (預防措施，基本用不到)
        X1 = int(np.clip(X1, 0, W-1))
        Y1 = int(np.clip(Y1, 0, H-1))
        X2 = int(np.clip(X2, 0, W-1))
        Y2 = int(np.clip(Y2, 0, H-1))

        outputs.append((float(conf), X1, Y1, X2, Y2))

    return outputs



img_paths = sorted(glob.glob(os.path.join(TEST_ROOT, "patient*", "*.png")))
print(f"共找到 {len(img_paths)} 張 testing 圖片")

with open(OUT_TXT, "w") as f:
    for img_path in img_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]

        preds = infer_single_image(img_path)

        for conf, x1, y1, x2, y2 in preds:
            f.write(f"{stem} 0 {conf:.5f} {x1} {y1} {x2} {y2}\n")

print("推論完成，已輸出：", OUT_TXT)
