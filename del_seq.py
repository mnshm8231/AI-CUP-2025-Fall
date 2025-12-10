import re
from collections import defaultdict

INPUT_FILE = "kfold_result/final_ensemble_roi.txt"

OUTPUT_FILTERED = "final_output.txt"       # 要上傳的最終結果

#以下兩個輸出只是用來觀察刪掉框的結果，比如刪掉的信心大小、數量、用以推測此策略的效果 可以無視
OUTPUT_REMOVED = "removed_output.txt"         # 被刪掉的（原始順序）
OUTPUT_REMOVED_SORTED = "removed_sorted_by_conf.txt"  

# 容忍缺少的 index 數量，用以確保漏檢不會讓整個序列被切斷
TOLERANCE = 1                   


def parse_patient_and_index(line: str):
    m = re.search(r'(patient\d+)_0*(\d+)', line)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def parse_conf(line: str):
    parts = line.strip().split()
    if len(parts) > 2:
        try:
            return float(parts[2])
        except:
            return -1.0
    return -1.0


def find_best_range(sorted_unique_indices, tol):
    """
    找出允許缺號 tol 的最長 [L, R] 區間
    """
    if not sorted_unique_indices:
        return None, None, 0, 0

    n = len(sorted_unique_indices)
    left = 0

    best_L = sorted_unique_indices[0]
    best_R = sorted_unique_indices[0]
    best_range_len = 1
    best_unique_cnt = 1

    for right in range(n):
        while left <= right:
            L = sorted_unique_indices[left]
            R = sorted_unique_indices[right]

            window_len = R - L + 1
            unique_cnt = right - left + 1
            missing = window_len - unique_cnt

            if missing > tol:
                left += 1
            else:
                if (window_len > best_range_len) or (
                    window_len == best_range_len and unique_cnt > best_unique_cnt
                ):
                    best_L = L
                    best_R = R
                    best_range_len = window_len
                    best_unique_cnt = unique_cnt
                break

    return best_L, best_R, best_range_len, best_unique_cnt


def main():
    # 分病患存
    patient_to_entries = defaultdict(list)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue

            pid, idx = parse_patient_and_index(raw)
            if pid is None:
                print("⚠ 無法解析 (略過)：", raw)
                continue

            conf = parse_conf(raw)

            patient_to_entries[pid].append((idx, conf, raw + "\n"))

    all_kept = []
    all_removed = []
    all_removed_for_sort = []  # (conf, line)

    print("=== 每個病患的序列統計 ===")

    for pid, entries in sorted(patient_to_entries.items()):
        indices = [e[0] for e in entries]
        unique_indices = sorted(set(indices))

        L, R, range_len, unique_cnt = find_best_range(unique_indices, TOLERANCE)

        if L is None:
            print(f"{pid}: 無資料")
            continue

        kept = []
        removed = []

        for idx, conf, line in entries:
            if L <= idx <= R:
                kept.append(line)
            else:
                removed.append(line)
                all_removed_for_sort.append((conf, line))

        original = len(entries)
        kept_num = len(kept)
        removed_num = len(removed)
        missing_inside = range_len - unique_cnt

        print(f"--- {pid} ---")
        print(f"最長序列：{L} ~ {R}   (長度 = {range_len})")
        print(f"原本筆數：{original}")
        print(f"保留筆數：{kept_num}")
        print(f"刪除筆數：{removed_num}")

        if missing_inside > 0:
            print(f"⚠ WARNING: 在 {L}~{R} 內缺 {missing_inside} 個 index，但在容忍 {TOLERANCE} 內")

        print()

        all_kept.extend(kept)
        all_removed.extend(removed)

    # === 寫檔：保留 ===
    with open(OUTPUT_FILTERED, "w", encoding="utf-8") as f:
        f.writelines(all_kept)

    # === 寫檔：刪除（原始順序） ===
    with open(OUTPUT_REMOVED, "w", encoding="utf-8") as f:
        f.writelines(all_removed)

    # === 寫檔：刪除（依 confidence 排序） ===
    all_removed_for_sort.sort(key=lambda x: x[0], reverse=True)  # conf desc
    with open(OUTPUT_REMOVED_SORTED, "w", encoding="utf-8") as f:
        for conf, line in all_removed_for_sort:
            f.write(line)

    # === 全域統計 ===
    total_original = len(all_kept) + len(all_removed)
    print("=== 全體統計 ===")
    print(f"全檔案原本總筆數：{total_original}")
    print(f"全部保留筆數：    {len(all_kept)}")
    print(f"全部刪除筆數：    {len(all_removed)}")
    print()
    print(f"✔ 保留：{OUTPUT_FILTERED}")
    print(f"✔ 刪除：{OUTPUT_REMOVED}")
    print(f"✔ 刪除(排序)：{OUTPUT_REMOVED_SORTED}")


if __name__ == "__main__":
    main()
