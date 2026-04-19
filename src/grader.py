"""
Module chấm điểm cho hệ thống chấm trắc nghiệm tự động (OMR).

Nhận ảnh đã nắn chỉnh → tự động phát hiện vùng đáp án qua anchor markers
→ phân đoạn bằng adaptive threshold → chấm điểm bằng z-score.

Đặc điểm nổi bật:
  - Dùng z-score để miễn nhiễm với tẩy xóa bẩn (dirty erase)
  - Auto-detect vùng đáp án qua anchor markers (không hardcode tọa độ)
  - Hỗ trợ layout 2 cột × 10 hàng = 20 câu (phiếu chuẩn THPT)
  - Xuất kết quả JSON chuẩn

Author: TV4 (refactored — auto-detect)
"""

import json
import os
import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List

from src.config import ZSCORE_THRESHOLD, MIN_FILL_RATIO


# Ký hiệu 4 lựa chọn
_CHOICES = ['A', 'B', 'C', 'D']


# ============================================================
# HÀM 1: TRÍCH XUẤT VÙNG ĐÁP ÁN (ROI)
# ============================================================

def extract_bubble_grid(warped_image: np.ndarray,
                        roi_x: int = 0,
                        roi_y: int = 0,
                        roi_width: Optional[int] = None,
                        roi_height: Optional[int] = None) -> np.ndarray:
    """
    Cắt vùng chứa các ô đáp án (ROI) từ ảnh đã nắn chỉnh.

    Args:
        warped_image: Ảnh nắn chỉnh, shape (H, W) hoặc (H, W, 3)
        roi_x, roi_y: Tọa độ góc trên-trái. Mặc định 0.
        roi_width, roi_height: Kích thước. None = lấy toàn bộ.

    Returns:
        Vùng ROI, cùng dtype với ảnh đầu vào.

    Raises:
        ValueError: Nếu ROI vượt quá giới hạn ảnh.
    """
    img_h, img_w = warped_image.shape[:2]

    if roi_width is None:
        roi_width = img_w - roi_x
    if roi_height is None:
        roi_height = img_h - roi_y

    if roi_x < 0 or roi_y < 0:
        raise ValueError(
            f"roi_x và roi_y phải >= 0, nhận được roi_x={roi_x}, roi_y={roi_y}"
        )

    # Clip để tránh vượt biên (chống lệch tâm)
    roi_width = min(roi_width, img_w - roi_x)
    roi_height = min(roi_height, img_h - roi_y)

    if roi_width <= 0 or roi_height <= 0:
        raise ValueError(
            f"ROI không hợp lệ: ({roi_x}, {roi_y}, {roi_width}, {roi_height}), "
            f"ảnh: {img_w}×{img_h}"
        )

    return warped_image[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]


# ============================================================
# HÀM 2: PHÂN ĐOẠN BUBBLE (THRESHOLDING)
# ============================================================

def segment_bubbles(grid_image: np.ndarray,
                    threshold_method: str = "adaptive") -> np.ndarray:
    """
    Chuyển ảnh vùng đáp án sang ảnh nhị phân (vùng tô = trắng).

    Args:
        grid_image: Ảnh đầu vào, có thể là màu hoặc xám.
        threshold_method: "adaptive" | "otsu" | "binary"

    Returns:
        Ảnh nhị phân (H, W), pixel trắng (255) = vùng được tô.

    Raises:
        ValueError: Nếu threshold_method không hợp lệ.
    """
    # Bước 1 — Chuyển xám nếu cần
    if grid_image.ndim == 3:
        gray = cv2.cvtColor(grid_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = grid_image.copy()

    # Bước 2 — Validate method
    valid_methods = ("adaptive", "otsu", "binary")
    if threshold_method not in valid_methods:
        raise ValueError(
            f"threshold_method không hợp lệ: '{threshold_method}'. "
            f"Các giá trị hợp lệ: {valid_methods}"
        )

    # Bước 3 — Áp dụng threshold
    if threshold_method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15, C=4
        )
    elif threshold_method == "otsu":
        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:  # "binary"
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Bước 4 — Morphological cleanup: loại nhiễu nhỏ
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return binary


# ============================================================
# HÀM NỘI BỘ: ĐỌC ĐÁP ÁN 1 CÂU (Z-SCORE)
# ============================================================

def _read_one_question(binary: np.ndarray,
                       y: int,
                       xs: list,
                       bw: int = 31,
                       bh: int = 27) -> str:
    """
    Đọc đáp án 1 câu từ ảnh nhị phân bằng z-score.

    Z-score miễn nhiễm với tẩy xóa bẩn:
    - Bubble bị tẩy → pixel mờ → count thấp
    - Z-score chuẩn hóa → chỉ bubble nổi bật nhất mới được chọn

    Args:
        binary: Ảnh nhị phân toàn tờ.
        y: Tọa độ y góc trên-trái của hàng bubble.
        xs: Danh sách tọa độ x của 4 bubble (A, B, C, D).
        bw, bh: Kích thước vùng đọc mỗi bubble.

    Returns:
        Đáp án ('A'–'D') hoặc '?' nếu không xác định.
    """
    img_h, img_w = binary.shape[:2]
    counts = []

    for x in xs:
        # Clip tọa độ vào giới hạn ảnh (chống cắt lệch)
        y1 = max(0, y)
        y2 = min(img_h, y + bh)
        x1 = max(0, x)
        x2 = min(img_w, x + bw)
        bubble_region = binary[y1:y2, x1:x2]
        counts.append(int(cv2.countNonZero(bubble_region)))

    arr = np.array(counts, dtype=float)
    std = arr.std()

    # Nếu std quá nhỏ → tất cả bubble gần nhau → không tô
    if std < 10:
        return '?'

    z_scores = (arr - arr.mean()) / std
    max_idx = int(np.argmax(z_scores))

    if z_scores[max_idx] >= ZSCORE_THRESHOLD:
        return _CHOICES[max_idx]

    # Fallback: kiểm tra tỉ lệ pixel tuyệt đối
    bubble_area = bw * bh
    if arr[max_idx] > bubble_area * MIN_FILL_RATIO:
        return _CHOICES[max_idx]

    return '?'


# ============================================================
# HÀM 3: PHÁT HIỆN LƯỚI BUBBLE TỰ ĐỘNG
# ============================================================

def phat_hien_luoi_bubble(binary: np.ndarray,
                          roi: Tuple[int, int, int, int],
                          num_questions: int = 20,
                          choices_per_question: int = 4,
                          gray_image: np.ndarray = None
                          ) -> Tuple[List[int], List[int], List[int]]:
    """
    Phát hiện vị trí lưới bubble bằng HoughCircles.

    Thuật toán:
    1. Dùng HoughCircles trên ảnh xám → phát hiện TẤT CẢ bubble (cả tô + trống)
    2. Lọc circles trong vùng ROI
    3. Phân cụm tâm circles theo y → hàng, theo x → cột
    4. Chia thành 2 nửa (cột trái + cột phải)

    Ưu điểm so với contour detection:
    - HoughCircles phát hiện đường viền tròn → bắt được cả bubble trống
    - Contour detection chỉ phát hiện vùng tô đen → bỏ sót bubble trống

    Args:
        binary: Ảnh nhị phân toàn tờ (cho chấm điểm sau này)
        roi: (x, y, w, h) vùng đáp án
        num_questions: Tổng số câu (mặc định 20)
        choices_per_question: Số lựa chọn (mặc định 4)
        gray_image: Ảnh xám gốc cho HoughCircles (nếu None → dùng binary)

    Returns:
        Tuple (col_left_xs, col_right_xs, row_ys)
    """
    rx, ry, rw, rh = roi
    rows_per_col = num_questions // 2  # 10 hàng mỗi cột

    # ── Bước 1: Phát hiện circles bằng HoughCircles ──
    if gray_image is not None:
        src = gray_image
    else:
        src = binary

    blur = cv2.GaussianBlur(src, (5, 5), 0)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT,
        dp=1.2,         # Tỉ lệ accumulator
        minDist=20,      # Khoảng cách tối thiểu giữa 2 tâm
        param1=50,       # Ngưỡng Canny cao
        param2=25,       # Ngưỡng accumulator (nhạy)
        minRadius=10,    # Bán kính tối thiểu bubble
        maxRadius=18     # Bán kính tối đa bubble
    )

    if circles is None:
        return _fallback_grid(rx, ry, rw, rh, rows_per_col, choices_per_question)

    circles = np.round(circles[0]).astype(int)

    # ── Bước 2: Lọc circles trong vùng ROI ──
    bubbles = []
    for c in circles:
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
        # Chỉ giữ circles nằm trong ROI đáp án (mở rộng ±10px)
        if cy < ry - 10 or cy > ry + rh + 10:
            continue
        if cx < rx - 10 or cx > rx + rw + 10:
            continue
        bubbles.append({'cx': float(cx), 'cy': float(cy), 'r': r})

    if len(bubbles) < rows_per_col * choices_per_question:
        return _fallback_grid(rx, ry, rw, rh, rows_per_col, choices_per_question)

    # ── Bước 3: Phân cụm y → hàng ──
    all_cy = sorted([b['cy'] for b in bubbles])
    row_centers = _cluster_1d(all_cy, min_gap=15)

    if len(row_centers) > rows_per_col:
        row_centers = row_centers[:rows_per_col]
    elif len(row_centers) < rows_per_col:
        return _fallback_grid(rx, ry, rw, rh, rows_per_col, choices_per_question)

    # ── Bước 4: Phân cụm x → cột, lọc cột phụ ──
    all_cx = sorted([b['cx'] for b in bubbles])
    col_centers = _cluster_1d(all_cx, min_gap=15)

    # Đếm bubble mỗi cột → loại cột có quá ít (anchor hoặc noise)
    col_counts = []
    for cc in col_centers:
        count = sum(1 for b in bubbles if abs(b['cx'] - cc) < 15)
        col_counts.append((cc, count))

    # Chỉ giữ cột có >= 50% số hàng
    min_count = max(3, int(rows_per_col * 0.5))
    filtered_cols = [cc for cc, cnt in col_counts if cnt >= min_count]

    if len(filtered_cols) < 2 * choices_per_question:
        col_counts.sort(key=lambda x: x[1], reverse=True)
        n_need = 2 * choices_per_question
        filtered_cols = sorted([cc for cc, _ in col_counts[:n_need]])

    col_centers = filtered_cols

    # Chia 2 nửa: tìm gap lớn nhất
    if len(col_centers) >= 2 * choices_per_question:
        gaps = [(col_centers[i+1] - col_centers[i], i)
                for i in range(len(col_centers) - 1)]
        max_gap_idx = max(gaps, key=lambda g: g[0])[1]
        left_cols = col_centers[:max_gap_idx + 1]
        right_cols = col_centers[max_gap_idx + 1:]
    elif len(col_centers) >= choices_per_question:
        mid = len(col_centers) // 2
        left_cols = col_centers[:mid]
        right_cols = col_centers[mid:]
    else:
        return _fallback_grid(rx, ry, rw, rh, rows_per_col, choices_per_question)

    # Lấy đúng 4 cột A, B, C, D mỗi nửa
    left_cols = left_cols[-choices_per_question:]
    right_cols = right_cols[-choices_per_question:]

    # ── Bước 5: Chuyển centroid → tọa độ góc trên-trái ──
    avg_r = int(np.mean([b['r'] for b in bubbles]))
    col_left_xs = [int(c - avg_r) for c in left_cols]
    col_right_xs = [int(c - avg_r) for c in right_cols]
    row_ys = [int(r - avg_r) for r in row_centers]

    return col_left_xs, col_right_xs, row_ys


def _cluster_1d(values: list, min_gap: float = 15) -> List[float]:
    """Phân cụm dãy số 1D: gộp các giá trị gần nhau (< min_gap) thành 1 cụm."""
    if not values:
        return []
    clusters = [[values[0]]]
    for v in values[1:]:
        if v - clusters[-1][-1] < min_gap:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [float(np.median(c)) for c in clusters]


def _fallback_grid(rx, ry, rw, rh, rows_per_col, choices_per_question):
    """Fallback: chia đều vùng ROI thành grid."""
    half_w = rw // 2
    row_spacing = rh // rows_per_col

    row_ys = [ry + int(row_spacing * i) for i in range(rows_per_col)]

    left_start = rx + int(half_w * 0.30)
    left_spacing = int(half_w * 0.60) // choices_per_question
    col_left_xs = [left_start + left_spacing * j for j in range(choices_per_question)]

    right_start = rx + half_w + int(half_w * 0.30)
    right_spacing = int(half_w * 0.60) // choices_per_question
    col_right_xs = [right_start + right_spacing * j for j in range(choices_per_question)]

    return col_left_xs, col_right_xs, row_ys


# ============================================================
# HÀM 4: CHẤM ĐIỂM
# ============================================================

def calculate_score(segmented_image: np.ndarray,
                    answer_key: Dict[int, str],
                    num_questions: int = 20,
                    choices_per_question: int = 4,
                    col_left_xs: list = None,
                    col_right_xs: list = None,
                    row_ys: list = None,
                    bubble_w: int = 31,
                    bubble_h: int = 27
                    ) -> Tuple[int, float, Dict[int, str]]:
    """
    Chấm điểm từ ảnh nhị phân, so sánh với đáp án chuẩn.

    Hỗ trợ 2 chế độ:
    - Layout tọa độ: dùng khi truyền col_left_xs, col_right_xs, row_ys
    - Layout tự động (grid đều): dùng khi các tham số = None

    Args:
        segmented_image: Ảnh nhị phân (H, W).
        answer_key: Đáp án chuẩn {1: 'A', 2: 'C', ...}.
        num_questions: Tổng số câu. Mặc định 20.
        choices_per_question: Số lựa chọn. Mặc định 4.
        col_left_xs, col_right_xs, row_ys: Tọa độ grid. None = chia đều.
        bubble_w, bubble_h: Kích thước vùng đọc mỗi bubble.

    Returns:
        Tuple (correct_count, score, student_answers)
    """
    student_answers: Dict[int, str] = {}
    correct_count = 0
    img_h, img_w = segmented_image.shape[:2]

    # Lọc answer_key: chỉ lấy câu <= num_questions
    answer_key = {k: v for k, v in answer_key.items() if k <= num_questions}

    # ── Chế độ 1: Layout tọa độ (từ auto-detect hoặc truyền vào) ──
    if col_left_xs is not None and col_right_xs is not None and row_ys is not None:
        rows_per_col = len(row_ys)
        for row_i, y in enumerate(row_ys):
            for col_group, (q_offset, xs) in enumerate(
                [(0, col_left_xs), (rows_per_col, col_right_xs)]
            ):
                q_num = row_i + 1 + q_offset
                if q_num > num_questions:
                    continue
                ans = _read_one_question(segmented_image, y, xs, bubble_w, bubble_h)
                student_answers[q_num] = ans
                if ans == answer_key.get(q_num):
                    correct_count += 1

    # ── Chế độ 2: Layout tự động (chia đều ảnh) ──
    else:
        bubble_height = img_h // num_questions
        bubble_width = img_w // choices_per_question
        bubble_area = bubble_width * bubble_height

        for i in range(num_questions):
            counts = []
            for j in range(choices_per_question):
                y1 = i * bubble_height
                y2 = (i + 1) * bubble_height
                x1 = j * bubble_width
                x2 = (j + 1) * bubble_width
                region = segmented_image[y1:y2, x1:x2]
                counts.append(int(cv2.countNonZero(region)))

            arr = np.array(counts, dtype=float)
            std = arr.std()
            q_num = i + 1

            if std < 10:
                student_answers[q_num] = '?'
                continue

            z_scores = (arr - arr.mean()) / std
            max_idx = int(np.argmax(z_scores))

            if z_scores[max_idx] >= ZSCORE_THRESHOLD:
                ans = _CHOICES[max_idx] if max_idx < len(_CHOICES) else '?'
            elif arr[max_idx] > bubble_area * MIN_FILL_RATIO:
                ans = _CHOICES[max_idx] if max_idx < len(_CHOICES) else '?'
            else:
                ans = '?'

            student_answers[q_num] = ans
            if ans == answer_key.get(q_num):
                correct_count += 1

    # Tính điểm thang 10
    total = num_questions
    score = round((correct_count / total) * 10, 2) if total > 0 else 0.0

    return correct_count, score, student_answers


# ============================================================
# HÀM 5: XUẤT KẾT QUẢ JSON
# ============================================================

def export_result_json(correct_count: int,
                       score: float,
                       student_answers: Dict[int, str],
                       answer_key: Dict[int, str],
                       image_path: str = "",
                       so_bao_danh: str = "N/A",
                       ma_de: str = "N/A") -> str:
    """
    Xuất kết quả chấm điểm ra chuỗi JSON chuẩn.

    Args:
        correct_count: Số câu đúng.
        score: Điểm số thang 10.
        student_answers: Đáp án học sinh.
        answer_key: Đáp án chuẩn.
        image_path: Đường dẫn ảnh gốc.
        so_bao_danh: Số báo danh đã đọc.
        ma_de: Mã đề đã đọc.

    Returns:
        Chuỗi JSON đã indent.
    """
    total = len(answer_key)
    details = {}
    for q_num in sorted(answer_key.keys()):
        student_ans = student_answers.get(q_num, '?')
        correct_ans = answer_key[q_num]
        details[str(q_num)] = {
            "student": student_ans,
            "correct": correct_ans,
            "result": "correct" if student_ans == correct_ans else "wrong"
        }

    # Tạo dict đáp án đọc được (cho tương thích format sample_answer_key)
    answers_dict = {}
    for q_num in sorted(student_answers.keys()):
        answers_dict[str(q_num)] = student_answers[q_num]

    result = {
        "image_path": image_path,
        "so_bao_danh": so_bao_danh,
        "ma_de": ma_de,
        "num_questions": total,
        "total": total,
        "correct": correct_count,
        "wrong": total - correct_count,
        "score": score,
        "score_display": f"{score:.2f}/10",
        "answers": answers_dict,
        "details": details
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


# ============================================================
# HÀM 6: IN BẢNG KẾT QUẢ RA TERMINAL
# ============================================================

def print_result_table(correct_count: int,
                       score: float,
                       student_answers: Dict[int, str],
                       answer_key: Dict[int, str],
                       so_bao_danh: str = "N/A",
                       ma_de: str = "N/A") -> None:
    """
    In bảng kết quả chấm điểm ra terminal.

    Args:
        correct_count: Số câu đúng.
        score: Điểm thang 10.
        student_answers: Đáp án học sinh.
        answer_key: Đáp án chuẩn.
        so_bao_danh: Số báo danh.
        ma_de: Mã đề.
    """
    total = len(answer_key)
    print("\n" + "=" * 52)
    print("   KẾT QUẢ CHẤM TRẮC NGHIỆM TỰ ĐỘNG (OMR)")
    print("=" * 52)
    print(f"   Số báo danh : {so_bao_danh}")
    print(f"   Mã đề       : {ma_de}")
    print(f"   Số câu đúng : {correct_count}/{total}")
    print(f"   Điểm số     : {score:.2f}/10")
    print("=" * 52)
    print(f"   {'Câu':>4}  {'Học sinh':^8}  {'Đáp án':^8}  {'Kết quả'}")
    print("   " + "-" * 46)

    for q_num in sorted(answer_key.keys()):
        student_ans = student_answers.get(q_num, '?')
        correct_ans = answer_key[q_num]
        status = "✓" if student_ans == correct_ans else "✗"
        mark = " ←" if student_ans != correct_ans else ""
        print(f"   {q_num:>4}  {student_ans:^8}  {correct_ans:^8}  {status}{mark}")

    print("=" * 52 + "\n")


# ============================================================
# HÀM 7: PIPELINE ĐẦY ĐỦ — AUTO-DETECT + CHẤM ĐIỂM
# ============================================================

def grade_from_image(warped_image: np.ndarray,
                     answer_key: Dict[int, str],
                     num_questions: int = 20,
                     save_json: str = None,
                     image_path: str = "",
                     so_bao_danh: str = "N/A",
                     ma_de: str = "N/A") -> Tuple[int, float, Dict[int, str]]:
    """
    Pipeline đầy đủ: ảnh nắn chỉnh → auto-detect grid → chấm điểm.

    Args:
        warped_image: Ảnh đã nắn chỉnh (output bước 6).
        answer_key: Đáp án chuẩn.
        num_questions: Số câu (mặc định 20).
        save_json: Đường dẫn file JSON output. None = không lưu.
        image_path: Đường dẫn ảnh gốc (để ghi vào JSON).
        so_bao_danh: SBD đã đọc từ reader.
        ma_de: Mã đề đã đọc từ reader.

    Returns:
        Tuple (correct_count, score, student_answers)
    """
    # Lọc answer_key chỉ lấy câu <= num_questions
    answer_key = {k: v for k, v in answer_key.items() if k <= num_questions}

    # Bước 1 — Auto-detect vùng đáp án qua anchor
    from src.reader import phat_hien_anchor, phan_loai_vung_roi

    try:
        anchors = phat_hien_anchor(warped_image)
        rois = phan_loai_vung_roi(anchors, *warped_image.shape[:2])
        roi_dap_an = rois['dap_an']
        print(f"   ✓ Auto-detect vùng đáp án: {roi_dap_an}")
    except ValueError as e:
        print(f"   ⚠️  Auto-detect thất bại: {e}")
        print(f"   → Dùng fallback tọa độ tham khảo")
        from src.config import ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT
        roi_dap_an = (ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT)

    # Bước 2 — Phân đoạn toàn ảnh
    if warped_image.ndim == 3:
        gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = warped_image.copy()

    binary = segment_bubbles(gray, threshold_method="adaptive")
    print(f"   ✓ Phân đoạn adaptive threshold — shape={binary.shape}")

    # Bước 3 — Phát hiện lưới bubble tự động (HoughCircles trên ảnh xám)
    col_left_xs, col_right_xs, row_ys = phat_hien_luoi_bubble(
        binary, roi_dap_an, num_questions=num_questions,
        gray_image=gray
    )
    print(f"   ✓ Phát hiện lưới: {len(row_ys)} hàng × 2 cột")

    # Bước 4 — Tính kích thước bubble phù hợp
    if len(row_ys) >= 2:
        bubble_h = max(1, int(abs(row_ys[1] - row_ys[0]) * 0.7))
    else:
        bubble_h = 27
    if len(col_left_xs) >= 2:
        bubble_w = max(1, int(abs(col_left_xs[1] - col_left_xs[0]) * 0.8))
    else:
        bubble_w = 31

    # Bước 5 — Chấm điểm
    correct_count, score, student_answers = calculate_score(
        binary,
        answer_key,
        num_questions=num_questions,
        col_left_xs=col_left_xs,
        col_right_xs=col_right_xs,
        row_ys=row_ys,
        bubble_w=bubble_w,
        bubble_h=bubble_h
    )

    # Bước 6 — In kết quả
    print_result_table(correct_count, score, student_answers, answer_key,
                       so_bao_danh=so_bao_danh, ma_de=ma_de)

    # Bước 7 — Xuất JSON nếu cần
    if save_json:
        json_str = export_result_json(
            correct_count, score, student_answers, answer_key,
            image_path=image_path, so_bao_danh=so_bao_danh, ma_de=ma_de
        )
        os.makedirs(os.path.dirname(save_json) or ".", exist_ok=True)
        with open(save_json, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"   ✓ Đã lưu kết quả JSON: {save_json}")

    return correct_count, score, student_answers