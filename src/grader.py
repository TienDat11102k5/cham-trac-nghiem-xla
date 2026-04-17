"""
Module chấm điểm cho hệ thống chấm trắc nghiệm tự động (OMR).

Nhận ảnh đã nắn chỉnh (output của TV3), trích xuất vùng đáp án,
phân đoạn bằng adaptive threshold, và tính điểm theo đáp án chuẩn.

Đặc điểm nổi bật:
  - Dùng z-score để miễn nhiễm với tẩy xóa bẩn (dirty erase)
  - Hỗ trợ layout 2 cột × 10 hàng = 20 câu (phiếu chuẩn THPT)
  - Xuất kết quả JSON chuẩn

Author: TV4
"""

import json
import cv2
import numpy as np
from typing import Dict, Tuple, Optional


# ============================================================
# CẤU HÌNH LAYOUT PHIẾU (khớp với ảnh nan_chinh 1200×800)
# ============================================================

# Tọa độ x tâm của 4 bubble (A, B, C, D) trong cột trái và phải
# Xác định bằng phân tích thực tế ảnh test_sheet_02_07_nan_chinh.jpg
_COL_LEFT_XS  = [192, 229, 265, 301]   # câu 1–10
_COL_RIGHT_XS = [415, 451, 487, 523]   # câu 11–20

# Tọa độ y tâm của 10 hàng câu hỏi
_ROW_YS = [811, 845, 881, 916, 952, 987, 1023, 1058, 1095, 1130]

# Kích thước vùng đọc cho mỗi bubble (px)
_BUBBLE_W = 31
_BUBBLE_H = 27

# Ký hiệu 4 lựa chọn
_CHOICES = ['A', 'B', 'C', 'D']

# Z-score tối thiểu để xác nhận 1 bubble được tô (chống tẩy xóa bẩn)
_ZSCORE_THRESHOLD = 1.4

# Tỉ lệ pixel tối thiểu so với diện tích bubble (fallback khi std ~ 0)
_MIN_FILL_RATIO = 0.15


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
        warped_image (np.ndarray): Ảnh nắn chỉnh, shape (H, W) hoặc (H, W, 3).
        roi_x (int): Tọa độ x góc trên-trái. Mặc định 0.
        roi_y (int): Tọa độ y góc trên-trái. Mặc định 0.
        roi_width (int|None): Chiều rộng ROI. None = lấy toàn bộ chiều rộng.
        roi_height (int|None): Chiều cao ROI. None = lấy toàn bộ chiều cao.

    Returns:
        np.ndarray: Vùng ROI, cùng dtype với ảnh đầu vào.

    Raises:
        ValueError: Nếu ROI vượt quá giới hạn ảnh.

    Examples:
        >>> grid = extract_bubble_grid(warped, roi_x=0, roi_y=750,
        ...                            roi_width=800, roi_height=450)
        >>> print(grid.shape)
        (450, 800, 3)
    """
    img_h, img_w = warped_image.shape[:2]

    # Gán giá trị mặc định nếu None
    if roi_width is None:
        roi_width = img_w - roi_x
    if roi_height is None:
        roi_height = img_h - roi_y

    # Kiểm tra ROI hợp lệ
    if roi_x < 0 or roi_y < 0:
        raise ValueError(
            f"roi_x và roi_y phải >= 0, nhận được roi_x={roi_x}, roi_y={roi_y}"
        )
    if roi_x + roi_width > img_w or roi_y + roi_height > img_h:
        raise ValueError(
            f"ROI ({roi_x}+{roi_width}={roi_x+roi_width}, "
            f"{roi_y}+{roi_height}={roi_y+roi_height}) "
            f"vượt quá kích thước ảnh ({img_w}×{img_h})"
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
        grid_image (np.ndarray): Ảnh đầu vào, có thể là màu hoặc xám.
        threshold_method (str): Phương pháp phân ngưỡng.
            - "adaptive" : cv2.adaptiveThreshold (mặc định, tốt nhất cho ảnh scan)
            - "otsu"     : Otsu's method (tự tìm ngưỡng tối ưu)
            - "binary"   : Ngưỡng cố định 127

    Returns:
        np.ndarray: Ảnh nhị phân shape (H, W), dtype uint8.
                    Pixel trắng (255) = vùng được tô.

    Raises:
        ValueError: Nếu threshold_method không hợp lệ.

    Examples:
        >>> binary = segment_bubbles(grid, threshold_method="adaptive")
    """
    # Bước 1 - Chuyển sang xám nếu cần
    if grid_image.ndim == 3:
        gray = cv2.cvtColor(grid_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = grid_image.copy()

    # Bước 2 - Validate method
    valid_methods = ("adaptive", "otsu", "binary")
    if threshold_method not in valid_methods:
        raise ValueError(
            f"threshold_method không hợp lệ: '{threshold_method}'. "
            f"Các giá trị hợp lệ: {valid_methods}"
        )

    # Bước 3 - Áp dụng threshold
    if threshold_method == "adaptive":
        # Adaptive Gaussian: tốt nhất cho ảnh scan có độ sáng không đều
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15,   # vùng lân cận 15×15 px
            C=4             # trừ hằng số 4 để loại nền sáng nhẹ
        )

    elif threshold_method == "otsu":
        # Otsu: tự tính ngưỡng dựa trên histogram
        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

    else:  # "binary"
        # Ngưỡng cố định
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Bước 4 - Morphological cleanup: loại nhiễu nhỏ, giữ nguyên bubble
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return binary


# ============================================================
# HÀM NỘI BỘ: ĐỌC ĐÁP ÁN 1 CÂU (Z-SCORE)
# ============================================================

def _read_one_question(binary: np.ndarray,
                       y: int,
                       xs: list,
                       bw: int = _BUBBLE_W,
                       bh: int = _BUBBLE_H) -> str:
    """
    Đọc đáp án 1 câu từ ảnh nhị phân bằng z-score.

    Thuật toán z-score miễn nhiễm với tẩy xóa bẩn:
    - Bubble bị tẩy → còn lại pixel mờ → count thấp hơn bubble mới
    - Z-score chuẩn hóa → chỉ bubble có count nổi bật nhất (z > threshold) mới được chọn

    Args:
        binary (np.ndarray): Ảnh nhị phân toàn tờ.
        y (int): Tọa độ y góc trên-trái của hàng bubble.
        xs (list[int]): Danh sách tọa độ x của 4 bubble (A, B, C, D).
        bw, bh (int): Kích thước vùng đọc mỗi bubble.

    Returns:
        str: Đáp án ('A'–'D') hoặc '?' nếu không xác định.
    """
    img_h, img_w = binary.shape[:2]
    counts = []

    for x in xs:
        # Clip để tránh vượt biên
        y1 = max(0, y)
        y2 = min(img_h, y + bh)
        x1 = max(0, x)
        x2 = min(img_w, x + bw)
        bubble_region = binary[y1:y2, x1:x2]
        counts.append(int(cv2.countNonZero(bubble_region)))

    arr = np.array(counts, dtype=float)
    std = arr.std()

    # Nếu std quá nhỏ → tất cả bubble gần nhau → không tô hoặc tẩy xóa không rõ
    if std < 10:
        return '?'

    # Z-score: chuẩn hóa
    z_scores = (arr - arr.mean()) / std
    max_idx = int(np.argmax(z_scores))

    if z_scores[max_idx] >= _ZSCORE_THRESHOLD:
        return _CHOICES[max_idx]

    # Fallback: kiểm tra tỉ lệ pixel tuyệt đối
    bubble_area = bw * bh
    if arr[max_idx] > bubble_area * _MIN_FILL_RATIO:
        return _CHOICES[max_idx]

    return '?'


# ============================================================
# HÀM 3: CHẤM ĐIỂM
# ============================================================

def calculate_score(segmented_image: np.ndarray,
                    answer_key: Dict[int, str],
                    num_questions: int = 20,
                    choices_per_question: int = 4,
                    col_left_xs: list = None,
                    col_right_xs: list = None,
                    row_ys: list = None,
                    bubble_w: int = _BUBBLE_W,
                    bubble_h: int = _BUBBLE_H
                    ) -> Tuple[int, float, Dict[int, str]]:
    """
    Chấm điểm từ ảnh nhị phân, so sánh với đáp án chuẩn.

    Hỗ trợ 2 chế độ layout:
    - Layout tự động (grid đều): dùng khi col_left_xs=None
      → chia đều segmented_image thành lưới num_questions × choices_per_question
    - Layout thực tế (toạ độ cứng): dùng khi truyền vào col_left_xs, col_right_xs, row_ys
      → khớp chính xác với tờ phiếu THPT chuẩn

    Args:
        segmented_image (np.ndarray): Ảnh nhị phân (H, W).
        answer_key (Dict[int, str]): Đáp án chuẩn {1: 'A', 2: 'C', ...}.
        num_questions (int): Tổng số câu. Mặc định 20.
        choices_per_question (int): Số lựa chọn. Mặc định 4.
        col_left_xs (list|None): Tọa độ x 4 bubble cột trái. None = auto.
        col_right_xs (list|None): Tọa độ x 4 bubble cột phải. None = auto.
        row_ys (list|None): Tọa độ y của mỗi hàng câu. None = auto.
        bubble_w, bubble_h (int): Kích thước vùng đọc mỗi bubble.

    Returns:
        Tuple[int, float, Dict[int, str]]:
            - correct_count: Số câu đúng.
            - score: Điểm thang 10 (làm tròn 2 chữ số).
            - student_answers: {1: 'B', 2: 'C', ...}

    Examples:
        >>> correct, score, answers = calculate_score(binary, answer_key)
        >>> print(f"Điểm: {score}/10")
    """
    student_answers: Dict[int, str] = {}
    correct_count = 0
    img_h, img_w = segmented_image.shape[:2]

    # ── Chế độ 1: Layout thực tế (tọa độ cứng từ phân tích ảnh) ──
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
        bubble_width  = img_w // choices_per_question
        bubble_area   = bubble_width * bubble_height

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

            if z_scores[max_idx] >= _ZSCORE_THRESHOLD:
                ans = _CHOICES[max_idx] if max_idx < len(_CHOICES) else '?'
            elif arr[max_idx] > bubble_area * _MIN_FILL_RATIO:
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
# HÀM 4: XUẤT KẾT QUẢ JSON
# ============================================================

def export_result_json(correct_count: int,
                       score: float,
                       student_answers: Dict[int, str],
                       answer_key: Dict[int, str],
                       image_path: str = "") -> str:
    """
    Xuất kết quả chấm điểm ra chuỗi JSON chuẩn.

    Args:
        correct_count (int): Số câu đúng.
        score (float): Điểm số thang 10.
        student_answers (Dict[int, str]): Đáp án học sinh.
        answer_key (Dict[int, str]): Đáp án chuẩn.
        image_path (str): Đường dẫn ảnh gốc (để truy xuất).

    Returns:
        str: Chuỗi JSON đã được indent.

    Examples:
        >>> json_str = export_result_json(15, 7.5, answers, key)
        >>> print(json_str)
    """
    total = len(answer_key)
    details = {}
    for q_num in sorted(answer_key.keys()):
        student_ans = student_answers.get(q_num, '?')
        correct_ans = answer_key[q_num]
        details[str(q_num)] = {
            "student":  student_ans,
            "correct":  correct_ans,
            "result":   "correct" if student_ans == correct_ans else "wrong"
        }

    result = {
        "image_path":    image_path,
        "total":         total,
        "correct":       correct_count,
        "wrong":         total - correct_count,
        "score":         score,
        "score_display": f"{score:.2f}/10",
        "details":       details
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


# ============================================================
# HÀM 5: IN BẢNG KẾT QUẢ RA TERMINAL
# ============================================================

def print_result_table(correct_count: int,
                       score: float,
                       student_answers: Dict[int, str],
                       answer_key: Dict[int, str]) -> None:
    """
    In bảng kết quả ra terminal theo định dạng dễ đọc.

    Args:
        correct_count (int): Số câu đúng.
        score (float): Điểm số thang 10.
        student_answers (Dict[int, str]): Đáp án học sinh.
        answer_key (Dict[int, str]): Đáp án chuẩn.
    """
    total = len(answer_key)
    print("\n" + "=" * 52)
    print("   KẾT QUẢ CHẤM TRẮC NGHIỆM TỰ ĐỘNG (OMR)")
    print("=" * 52)
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
# CHẠY TRỰC TIẾP: pipeline demo với ảnh thật từ TV3
# ============================================================

def grade_from_image(image_path: str,
                     answer_key: Dict[int, str],
                     save_json: str = None) -> Tuple[int, float, Dict[int, str]]:
    """
    Pipeline đầy đủ: đọc ảnh nan_chinh → phân đoạn → chấm điểm → in kết quả.

    Args:
        image_path (str): Đường dẫn ảnh đã nắn chỉnh (output TV3).
        answer_key (Dict[int, str]): Đáp án chuẩn.
        save_json (str|None): Đường dẫn file JSON đầu ra. None = không lưu.

    Returns:
        Tuple[int, float, Dict[int, str]]: (correct_count, score, student_answers)
    """
    import os

    # 1. Đọc ảnh
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy file ảnh: '{image_path}'")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: '{image_path}'")

    print(f"✓ Đọc ảnh: {image_path} — shape={img.shape}")

    # 2. Phân đoạn toàn ảnh
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = segment_bubbles(gray, threshold_method="adaptive")
    print(f"✓ Phân đoạn adaptive threshold — shape={binary.shape}")
    # 🔥 THÊM DÒNG NÀY
    answer_key = {k: v for k, v in answer_key.items() if k <= 20}



    # 3. Chấm điểm (dùng layout tọa độ cứng khớp với phiếu thật)
    correct_count, score, student_answers = calculate_score(
        binary,
        answer_key,
        num_questions=20,
        col_left_xs=_COL_LEFT_XS,
        col_right_xs=_COL_RIGHT_XS,
        row_ys=_ROW_YS,
        bubble_w=_BUBBLE_W,
        bubble_h=_BUBBLE_H
    )

    # 4. In kết quả
    print_result_table(correct_count, score, student_answers, answer_key)

    # 5. Xuất JSON nếu cần
    if save_json:
        json_str = export_result_json(
            correct_count, score, student_answers, answer_key, image_path
        )
        with open(save_json, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"✓ Đã lưu kết quả JSON: {save_json}")

    return correct_count, score, student_answers


if __name__ == "__main__":
    # ── Demo chạy thẳng với ảnh test từ TV3 ──
    import json as _json
    import sys

    # Đường dẫn ảnh nắn chỉnh (output TV3)
    IMAGE_PATH = "data/processed/test_sheet_02_07_nan_chinh.jpg"

    # Đọc đáp án chuẩn
    KEY_PATH = "data/answer_keys/sample_answer_key.json"
    with open(KEY_PATH, "r", encoding="utf-8") as f:
        raw = _json.load(f)
    answer_key = {int(k): v for k, v in raw["answers"].items()}

    grade_from_image(IMAGE_PATH, answer_key, save_json="data/result_output.json")