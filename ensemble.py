import os
from collections import defaultdict
import numpy as np


# 每個 fold 的輸入檔（順序隨意，但會當成 fold_id）
# 無須重跑推論，已準備好五個 fold 的結果檔

FOLD_FILES = [
    "kfold_result/ensmble_roi_1.txt",
    "kfold_result/ensmble_roi_2.txt",
    "kfold_result/ensmble_roi_3.txt",
    "kfold_result/ensmble_roi_4.txt",
    "kfold_result/ensmble_roi_5.txt",
]

# 輸出 ensemble 結果
OUT_FILE = "kfold_result/final_ensemble_roi.txt"

IOU_THR = 0.5

# 至少多少不同 fold 都有投票，才保留這個 cluster
MIN_FOLDS = 2 

#   conf ≥ MIN_CONF_JOIN 才參與 cluster & 加權
#   conf < MIN_CONF_JOIN 的框當作噪音，完全忽略
MIN_CONF_JOIN = 0.001   



# IoU 計算
def bbox_iou(box1, box2):
    """
    box1, box2: [x1, y1, x2, y2]
    回傳 IoU (float)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    if inter <= 0:
        return 0.0

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - inter
    if union <= 0:
        return 0.0

    return inter / union



# 讀取所有 folds
def load_all_folds(fold_files):
    img_to_boxes = defaultdict(list)

    for fold_id, file_path in enumerate(fold_files):
        if not os.path.exists(file_path):
            print(f"⚠ 找不到檔案，略過: {file_path}")
            continue

        print(f"📥 讀取 fold {fold_id} 檔案: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 7:
                    continue

                img_id = parts[0]
                try:
                    cls_id = int(float(parts[1]))  
                    conf = float(parts[2])
                    x1 = float(parts[3])
                    y1 = float(parts[4])
                    x2 = float(parts[5])
                    y2 = float(parts[6])
                except ValueError:
                    print(f"⚠ 無法解析數值，略過: {line}")
                    continue

                img_to_boxes[img_id].append(
                    {
                        "cls": cls_id,
                        "conf": conf,
                        "box": [x1, y1, x2, y2],
                        "fold": fold_id,
                    }
                )

    print(f"✅ 共讀到 {len(img_to_boxes)} 張圖片的預測")
    return img_to_boxes



# 對單張圖做 ensemble
def ensemble_one_image(box_list, iou_thr=0.5, min_folds=2, min_conf_join=0.01):

    if not box_list:
        return []

    high_conf_boxes = [b for b in box_list if b["conf"] >= min_conf_join]
    if not high_conf_boxes:
        return []  

    # 依 conf 由大到小，讓高信心框優先建立 cluster
    high_conf_boxes = sorted(high_conf_boxes, key=lambda x: x["conf"], reverse=True)

    clusters = []  # 每個元素: {'boxes': [...], 'folds': set(), 'rep_box': [x1,y1,x2,y2]}

    for entry in high_conf_boxes:
        b = entry["box"]
        fold_id = entry["fold"]

        best_cluster_idx = -1
        best_iou = 0.0

        # 找到 IoU 最大且大於門檻的 cluster
        for ci, c in enumerate(clusters):
            iou = bbox_iou(b, c["rep_box"])
            if iou > best_iou and iou >= iou_thr:
                best_iou = iou
                best_cluster_idx = ci

        if best_cluster_idx == -1:
            # 建立新 cluster
            clusters.append(
                {
                    "boxes": [entry],
                    "folds": {fold_id},
                    "rep_box": b[:],  # 拷貝
                }
            )
        else:
            # 加到既有 cluster
            c = clusters[best_cluster_idx]
            c["boxes"].append(entry)
            c["folds"].add(fold_id)

            # 更新該 cluster 的代表框 (用 conf 加權平均座標)
            boxes_arr = np.array([e["box"] for e in c["boxes"]], dtype=float)
            confs_arr = np.array([e["conf"] for e in c["boxes"]], dtype=float)
            w = confs_arr / (confs_arr.sum() + 1e-9)
            rep = (boxes_arr * w[:, None]).sum(axis=0)
            c["rep_box"] = rep.tolist()

    # 根據 cluster 的 folds 數量過濾
    final_boxes = []
    for c in clusters:
        if len(c["folds"]) < min_folds:
            continue

        boxes_arr = np.array([e["box"] for e in c["boxes"]], dtype=float)
        confs_arr = np.array([e["conf"] for e in c["boxes"]], dtype=float)
        cls_ids = [e["cls"] for e in c["boxes"]]

        # 座標用 conf 加權平均
        w = confs_arr / (confs_arr.sum() + 1e-9)
        avg_box = (boxes_arr * w[:, None]).sum(axis=0)

        # conf 用最大值
        out_conf = float(confs_arr.max())
        values, counts = np.unique(cls_ids, return_counts=True)
        out_cls = int(values[counts.argmax()])

        final_boxes.append(
            {
                "cls": out_cls,
                "conf": out_conf,
                "box": avg_box.tolist(),
            }
        )

    return final_boxes



# 跑整體 ensemble 並寫檔
def run_ensemble():
    img_to_boxes = load_all_folds(FOLD_FILES)

    total_before = 0
    total_after = 0
    total_low_conf_dropped = 0

    with open(OUT_FILE, "w", encoding="utf-8") as f_out:
        for img_id, box_list in sorted(img_to_boxes.items()):
            total_before += len(box_list)

            low_conf_cnt = sum(b["conf"] < MIN_CONF_JOIN for b in box_list)
            total_low_conf_dropped += low_conf_cnt

            ensembled = ensemble_one_image(
                box_list,
                iou_thr=IOU_THR,
                min_folds=MIN_FOLDS,
                min_conf_join=MIN_CONF_JOIN,
            )

            total_after += len(ensembled)

            for fb in ensembled:
                x1, y1, x2, y2 = fb["box"]
                cls_id = fb["cls"]
                conf = fb["conf"]
                x1_i = int(round(x1))
                y1_i = int(round(y1))
                x2_i = int(round(x2))
                y2_i = int(round(y2))

                line = f"{img_id} {cls_id} {conf:.5f} {x1_i} {y1_i} {x2_i} {y2_i}\n"
                f_out.write(line)

    print("=== Ensemble 完成 ===")
    print(f"原本總框數：{total_before}")
    print(f"ensemble 後總框數：{total_after}")
    print(f"輸出：{OUT_FILE}")


if __name__ == "__main__":
    run_ensemble()
