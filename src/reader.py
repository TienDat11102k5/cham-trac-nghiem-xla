"""
Module đọc thông tin từ phiếu trắc nghiệm (OMR Reader).

Module này chứa các hàm để tự động phát hiện và đọc:
- Số báo danh (SBD) — student ID
- Mã đề thi — exam code
- Vùng đáp án — answer region

Kỹ thuật chính:
- Anchor-based ROI Detection: Dùng ô vuông đen làm mốc tính tọa độ tự động
- Z-score: Đọc bubble chính xác, miễn nhiễm tẩy xóa bẩn
- Contour Analysis: Phân loại anchor vs bubble qua extent + solidity

Thành viên phụ trách: TV4 (refactored)
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict

from src.config import (
    ANCHOR_MIN_AREA, ANCHOR_MAX_AREA, ANCHOR_MIN_EXTENT,
    ANCHOR_MIN_SOLIDITY, ZSCORE_THRESHOLD, MIN_FILL_RATIO,
)


# ============================================================================
# HÀM PHÁT HIỆN ANCHOR MARKERS (Cốt lõi auto-detect)
# ============================================================================

def phat_hien_anchor(warped_image: np.ndarray) -> List[Dict]:
    """
    Phát hiện các ô vuông đen (anchor markers) trên ảnh đã nắn chỉnh.

    Thuật toán:
    1. Chuyển xám + Otsu threshold → ảnh nhị phân
    2. Morphological close → nối contour bị đứt
    3. findContours → tìm tất cả contour
    4. Lọc theo: diện tích, extent (> 0.82), solidity (> 0.93)
    5. Trả về danh sách anchor đã sắp xếp

    Args:
        warped_image: Ảnh đã nắn chỉnh, shape (H, W) hoặc (H, W, 3)

    Returns:
        Danh sách dict, mỗi dict chứa:
        - 'x', 'y', 'w', 'h': Bounding rect
        - 'cx', 'cy': Tâm (centroid) — chính xác sub-pixel
        - 'area', 'extent', 'solidity': Thuộc tính hình học

    Raises:
        ValueError: Nếu không tìm thấy anchor nào
    """
    # Bước 1 — Chuyển xám
    if warped_image.ndim == 3:
        gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = warped_image.copy()

    # Bước 2 — Otsu threshold (tách foreground/background tự động)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Bước 3 — Morphological close: nối contour đứt do nhiễu
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Bước 4 — Tìm tất cả contour
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Bước 5 — Lọc anchor theo 3 tiêu chí: area + extent + solidity
    anchors = []
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Lọc diện tích
        if area < ANCHOR_MIN_AREA or area > ANCHOR_MAX_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h

        # Lọc extent (tỉ lệ area / bounding_rect)
        # Anchor vuông đặc: extent > 0.82 | Bubble tròn: extent < 0.76
        extent = area / rect_area if rect_area > 0 else 0
        if extent < ANCHOR_MIN_EXTENT:
            continue

        # Lọc solidity (tỉ lệ area / convex_hull_area)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < ANCHOR_MIN_SOLIDITY:
            continue

        # Tính tâm chính xác bằng moments (sub-pixel)
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx = x + w / 2.0
            cy = y + h / 2.0

        anchors.append({
            'x': x, 'y': y, 'w': w, 'h': h,
            'cx': cx, 'cy': cy,
            'area': area, 'extent': extent, 'solidity': solidity
        })

    if not anchors:
        raise ValueError(
            "Không tìm thấy anchor markers trên phiếu. "
            "Kiểm tra: Phiếu có ô vuông đen ở các góc không?"
        )

    # Sắp xếp theo y rồi x (từ trên xuống, trái sang phải)
    anchors.sort(key=lambda a: (a['cy'], a['cx']))

    return anchors


def phan_loai_vung_roi(anchors: List[Dict],
                       img_h: int = 1200,
                       img_w: int = 800) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Từ danh sách anchor → phân loại và tính tọa độ 3 vùng ROI tự động.

    Phân tích ảnh reference chuẩn (test_sheet_01_roi_visualization.jpg):
    - SBD (xanh dương): bao header "SỐ BÁO DANH" + bảng bubble 6 cột × 10 hàng
    - Mã đề (xanh lá): bao header "MÃ ĐỀ" + bảng bubble 3 cột × 10 hàng  
    - Đáp án (đỏ): bao cả 2 cột (câu 1-10 + câu 11-20), bắt đầu từ câu 1

    Layout anchor trên ảnh 800×1200:
    - Anchor phân cách: y≈403, x≈388 (giữa SBD và Mã đề)
    - Anchor hàng trên đáp án: y≈789 (x≈53, 166, 388, 610)
    - Anchor hàng dưới đáp án: y≈1180 (x≈388)

    Args:
        anchors: Danh sách anchor từ phat_hien_anchor()
        img_h: Chiều cao ảnh
        img_w: Chiều rộng ảnh

    Returns:
        Dict với 3 key:
        - 'sbd': (x, y, w, h) — vùng Số Báo Danh
        - 'ma_de': (x, y, w, h) — vùng Mã Đề
        - 'dap_an': (x, y, w, h) — vùng Đáp Án
    """
    # Bước 1 — Phân chia theo y: trên (< 60% ảnh) và dưới (>= 60%)
    y_threshold = img_h * 0.60
    nhom_tren = [a for a in anchors if a['cy'] < y_threshold]
    nhom_duoi = [a for a in anchors if a['cy'] >= y_threshold]

    roi_sbd = None
    roi_ma_de = None

    # Bước 2 — Tính ROI cho SBD và Mã Đề
    if nhom_tren:
        nhom_tren_sorted = sorted(nhom_tren, key=lambda a: a['cx'])

        # Tìm anchor phân cách (x gần giữa ảnh, khoảng x/2)
        mid_x = img_w * 0.45
        phan_cach = [a for a in nhom_tren_sorted
                     if abs(a['cx'] - mid_x) < img_w * 0.15]

        if phan_cach:
            sep = phan_cach[0]

            # ── Y START: Ngang hàng header "SỐ BÁO DANH" / "MÃ ĐỀ"
            # Từ ảnh reference: vùng SBD bao cả header → y_start = anchor_y - offset
            # Header nằm ~ 2 lần chiều cao anchor phía trên anchor phân cách
            y_start = int(sep['cy'] - sep['h'] * 3.5)
            y_start = max(0, y_start)

            # ── Y END: Lấy từ anchor hàng trên nhóm dưới (ranh giới đáp án)
            if nhom_duoi:
                nhom_duoi_sorted_y = sorted(nhom_duoi, key=lambda a: a['cy'])
                y_end = int(nhom_duoi_sorted_y[0]['cy'] - nhom_duoi_sorted_y[0]['h'])
            else:
                y_end = int(img_h * 0.65)

            # ── SBD: từ cạnh trái ảnh đến anchor phân cách
            x_start_sbd = int(img_w * 0.10)
            x_end_sbd = int(sep['cx'] - sep['w'] * 0.3)

            roi_sbd = (x_start_sbd, y_start,
                       x_end_sbd - x_start_sbd, y_end - y_start)

            # ── Mã đề: từ anchor phân cách đến bên phải
            x_start_md = int(sep['cx'] - sep['w'] * 0.3)
            x_end_md = int(img_w * 0.72)

            roi_ma_de = (x_start_md, y_start,
                         x_end_md - x_start_md, y_end - y_start)

    # Bước 3 — Tính vùng đáp án từ nhóm dưới
    # Theo ảnh reference: vùng đỏ bao từ câu 1 đến câu 20 (cả 2 cột)
    roi_dap_an = None

    if len(nhom_duoi) >= 2:
        nhom_duoi_by_y = sorted(nhom_duoi, key=lambda a: a['cy'])

        # Chia anchor thành hàng trên (ranh giới trên) và hàng dưới (ranh giới dưới)
        y_mid_duoi = (nhom_duoi_by_y[0]['cy'] + nhom_duoi_by_y[-1]['cy']) / 2
        hang_tren_da = [a for a in nhom_duoi_by_y if a['cy'] < y_mid_duoi]
        hang_duoi_da = [a for a in nhom_duoi_by_y if a['cy'] >= y_mid_duoi]

        if hang_tren_da and hang_duoi_da:
            # Kiểm tra xem hàng trên và hàng dưới có thực sự khác hàng không?
            # Nếu ảnh bị cắt mất phần dưới (như test_sheet_03), 
            # 2 anchor cùng hàng trên sẽ bị ép chia vào hang_tren_da và hang_duoi_da.
            y_diff = hang_duoi_da[-1]['cy'] - hang_tren_da[0]['cy']
            
            # X: lấy anchor xa nhất trái/phải → mở rộng thêm
            all_x = [a['cx'] for a in nhom_duoi]
            x_min = int(min(all_x) - 15)
            x_max = int(max(all_x) + 15)

            # Y top: Ngay dưới hàng anchor trên (+ offset header "A B C D")
            anchor_h_avg = np.mean([a['h'] for a in hang_tren_da])
            y_top = int(hang_tren_da[0]['cy'] + anchor_h_avg * 1.6)

            if y_diff >= anchor_h_avg * 3:
                # Ảnh bình thường: có cả hàng trên và hàng dưới
                y_bot = int(hang_duoi_da[-1]['cy'] + hang_duoi_da[-1]['h'])
            else:
                # Ảnh bị crop mất hàng dưới: quét tới gần cuối ảnh
                y_bot = int(img_h * 0.98)

            roi_dap_an = (x_min, y_top, x_max - x_min, y_bot - y_top)
        else:
            all_x = [a['cx'] for a in nhom_duoi]
            all_y = [a['cy'] for a in nhom_duoi]
            x_min = int(min(all_x) - 15)
            x_max = int(max(all_x) + 15)
            y_min = int(min(all_y))
            y_max = int(img_h * 0.98) # Mở rộng xuống cuối nếu chỉ có 1 hàng
            roi_dap_an = (x_min, y_min, x_max - x_min, y_max - y_min)

    # Bước 4 — Fallback nếu auto-detect thất bại
    from src.config import (
        EXAM_CODE_ROI_X, EXAM_CODE_ROI_Y, EXAM_CODE_ROI_WIDTH, EXAM_CODE_ROI_HEIGHT,
        STUDENT_ID_ROI_X, STUDENT_ID_ROI_Y, STUDENT_ID_ROI_WIDTH, STUDENT_ID_ROI_HEIGHT,
        ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT,
    )

    if roi_sbd is None:
        roi_sbd = (STUDENT_ID_ROI_X, STUDENT_ID_ROI_Y,
                   STUDENT_ID_ROI_WIDTH, STUDENT_ID_ROI_HEIGHT)

    if roi_ma_de is None:
        roi_ma_de = (EXAM_CODE_ROI_X, EXAM_CODE_ROI_Y,
                     EXAM_CODE_ROI_WIDTH, EXAM_CODE_ROI_HEIGHT)

    if roi_dap_an is None:
        roi_dap_an = (ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT)

    return {
        'sbd': roi_sbd,
        'ma_de': roi_ma_de,
        'dap_an': roi_dap_an,
    }


# ============================================================================
# HÀM CẮT VÙNG ROI (Hỗ trợ auto-detect + manual)
# ============================================================================

def extract_exam_code_region(warped_image: np.ndarray,
                             roi_x: int = None,
                             roi_y: int = None,
                             roi_width: int = None,
                             roi_height: int = None) -> np.ndarray:
    """
    Cắt vùng chứa mã đề thi từ ảnh đã nắn chỉnh.

    Nếu không truyền tọa độ → tự động phát hiện qua anchor markers.
    Nếu truyền tọa độ → dùng tọa độ truyền vào.

    Args:
        warped_image: Ảnh đã nắn chỉnh
        roi_x, roi_y, roi_width, roi_height: Tọa độ thủ công (None = auto)

    Returns:
        Ảnh vùng ROI chứa mã đề

    Raises:
        ValueError: Nếu ROI vượt quá giới hạn ảnh
    """
    # Auto-detect nếu không truyền tọa độ
    if roi_x is None or roi_y is None or roi_width is None or roi_height is None:
        anchors = phat_hien_anchor(warped_image)
        rois = phan_loai_vung_roi(anchors, *warped_image.shape[:2])
        roi_x, roi_y, roi_width, roi_height = rois['ma_de']

    return _cat_vung_roi(warped_image, roi_x, roi_y, roi_width, roi_height)


def extract_student_id_region(warped_image: np.ndarray,
                              roi_x: int = None,
                              roi_y: int = None,
                              roi_width: int = None,
                              roi_height: int = None) -> np.ndarray:
    """
    Cắt vùng chứa số báo danh (SBD) từ ảnh đã nắn chỉnh.

    Nếu không truyền tọa độ → tự động phát hiện qua anchor markers.

    Args:
        warped_image: Ảnh đã nắn chỉnh
        roi_x, roi_y, roi_width, roi_height: Tọa độ thủ công (None = auto)

    Returns:
        Ảnh vùng ROI chứa SBD

    Raises:
        ValueError: Nếu ROI vượt quá giới hạn ảnh
    """
    if roi_x is None or roi_y is None or roi_width is None or roi_height is None:
        anchors = phat_hien_anchor(warped_image)
        rois = phan_loai_vung_roi(anchors, *warped_image.shape[:2])
        roi_x, roi_y, roi_width, roi_height = rois['sbd']

    return _cat_vung_roi(warped_image, roi_x, roi_y, roi_width, roi_height)


def extract_answer_region(warped_image: np.ndarray,
                          roi_x: int = None,
                          roi_y: int = None,
                          roi_width: int = None,
                          roi_height: int = None) -> np.ndarray:
    """
    Cắt vùng chứa đáp án từ ảnh đã nắn chỉnh.

    Args:
        warped_image: Ảnh đã nắn chỉnh
        roi_x, roi_y, roi_width, roi_height: Tọa độ thủ công (None = auto)

    Returns:
        Ảnh vùng ROI chứa đáp án
    """
    if roi_x is None or roi_y is None or roi_width is None or roi_height is None:
        anchors = phat_hien_anchor(warped_image)
        rois = phan_loai_vung_roi(anchors, *warped_image.shape[:2])
        roi_x, roi_y, roi_width, roi_height = rois['dap_an']

    return _cat_vung_roi(warped_image, roi_x, roi_y, roi_width, roi_height)


def _cat_vung_roi(image: np.ndarray,
                  x: int, y: int, w: int, h: int) -> np.ndarray:
    """
    Cắt vùng ROI với kiểm tra biên an toàn.

    Args:
        image: Ảnh nguồn
        x, y, w, h: Tọa độ và kích thước vùng cắt

    Returns:
        Ảnh vùng đã cắt

    Raises:
        ValueError: Nếu tọa độ không hợp lệ hoặc vượt biên
    """
    img_h, img_w = image.shape[:2]

    # Clip tọa độ vào giới hạn ảnh (chống lệch tâm)
    x = max(0, int(x))
    y = max(0, int(y))
    w = min(int(w), img_w - x)
    h = min(int(h), img_h - y)

    if w <= 0 or h <= 0:
        raise ValueError(
            f"ROI không hợp lệ sau khi clip: ({x}, {y}, {w}, {h}), "
            f"ảnh: {img_w}×{img_h}"
        )

    return image[y:y + h, x:x + w]


# ============================================================================
# HÀM ĐỌC MÃ ĐỀ / SBD (Z-score — chống tẩy xóa bẩn)
# ============================================================================

def read_exam_code(exam_code_region: np.ndarray,
                   num_digits: int = 3,
                   choices_per_digit: int = 10,
                   threshold_method: str = "otsu") -> str:
    """
    Đọc mã đề từ vùng ROI đã cắt, sử dụng HoughCircles + z-score.

    Thuật toán:
    1. HoughCircles → phát hiện TẤT CẢ bubble (cả tô + trống)
    2. Phân cụm x → cột (mỗi cột = 1 chữ số)
    3. Phân cụm y → hàng (0-9)
    4. Đọc pixel tô đậm → z-score → chọn chữ số nổi bật nhất

    Ưu điểm so chia đều:
    - Tự bỏ qua header ("SỐ BÁO DANH", "MÃ ĐỀ")
    - Không phụ thuộc ROI cắt chính xác

    Args:
        exam_code_region: Ảnh vùng chứa mã đề hoặc SBD
        num_digits: Số chữ số (3 cho mã đề, 6 cho SBD)
        choices_per_digit: Số lựa chọn mỗi chữ số (0-9 = 10)
        threshold_method: Phương pháp phân ngưỡng

    Returns:
        Mã đề/SBD dạng string (VD: "101", "012356")

    Raises:
        ValueError: Nếu không đọc được
    """
    # Bước 1 — Tiền xử lý
    if exam_code_region.ndim == 3:
        gray = cv2.cvtColor(exam_code_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = exam_code_region.copy()

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Bước 2 — Phân ngưỡng (cho bước đọc pixel sau)
    binary = _phan_nguong(gray, threshold_method)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Bước 3 — HoughCircles phát hiện bubble
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=15,
        param1=50, param2=25,
        minRadius=8, maxRadius=16
    )

    if circles is None or len(circles[0]) < num_digits * choices_per_digit * 0.5:
        # Fallback: dùng phương pháp chia đều cũ
        return _read_code_grid_divide(binary, num_digits, choices_per_digit)

    circles = np.round(circles[0]).astype(int)

    # Bước 4 — Phân cụm x → cột (mỗi cột = 1 chữ số)
    all_cx = sorted([int(c[0]) for c in circles])
    col_centers = _cluster_1d_simple(all_cx, min_gap=15)

    # Bước 5 — Phân cụm y → hàng (0-9)
    all_cy = sorted([int(c[1]) for c in circles])
    row_centers = _cluster_1d_simple(all_cy, min_gap=15)

    # Lọc: chỉ giữ cột có đủ bubble (≥ 50% số hàng)
    col_counts = []
    for cc in col_centers:
        count = sum(1 for c in circles if abs(int(c[0]) - cc) < 15)
        col_counts.append((cc, count))

    min_bubbles = max(3, int(len(row_centers) * 0.5))
    valid_cols = [(cc, cnt) for cc, cnt in col_counts if cnt >= min_bubbles]

    if len(valid_cols) < num_digits:
        # Nếu không đủ, lấy top theo count
        col_counts.sort(key=lambda x: x[1], reverse=True)
        valid_cols = col_counts[:num_digits]

    # Sắp xếp cột từ trái → phải
    valid_cols = sorted(valid_cols, key=lambda x: x[0])
    digit_cols = [cc for cc, cnt in valid_cols[:num_digits]]

    # Chỉ giữ đúng số hàng cần thiết (10)
    # Lấy N hàng CUỐI — vì hàng header nằm ở trên cùng
    if len(row_centers) > choices_per_digit:
        row_centers = row_centers[-choices_per_digit:]

    if len(row_centers) < choices_per_digit:
        return _read_code_grid_divide(binary, num_digits, choices_per_digit)

    # Validate: khoảng cách giữa các hàng phải tương đối đều
    # Nếu hàng đầu quá xa → có thể vẫn là header
    if len(row_centers) >= 3:
        gaps = [row_centers[i+1] - row_centers[i] for i in range(len(row_centers)-1)]
        median_gap = sorted(gaps)[len(gaps)//2]
        # Loại bỏ hàng đầu nếu gap đầu > 1.8x median (header lạc)
        while len(row_centers) > choices_per_digit:
            first_gap = row_centers[1] - row_centers[0]
            if first_gap > median_gap * 1.8:
                row_centers = row_centers[1:]
            else:
                break
        # Nếu vẫn dư → cắt đầu
        if len(row_centers) > choices_per_digit:
            row_centers = row_centers[-choices_per_digit:]

    # Bước 6 — Đọc từng chữ số bằng z-score
    avg_r = int(np.mean([int(c[2]) for c in circles]))
    bw = avg_r * 2 + 4  # Kích thước vùng đọc
    bh = avg_r * 2 + 4

    exam_code = ""
    img_h, img_w = binary.shape[:2]

    for digit_idx, col_x in enumerate(digit_cols):
        counts = []
        for row_idx, row_y in enumerate(row_centers):
            # Tâm bubble → góc trên-trái
            x1 = max(0, int(col_x - bw // 2))
            y1 = max(0, int(row_y - bh // 2))
            x2 = min(img_w, x1 + bw)
            y2 = min(img_h, y1 + bh)
            region = binary[y1:y2, x1:x2]
            counts.append(int(cv2.countNonZero(region)))

        # Z-score
        arr = np.array(counts, dtype=float)
        std = arr.std()

        if std < 10:
            raise ValueError(
                f"Chữ số thứ {digit_idx + 1}: Không phân biệt được ô nào được tô. "
                f"Pixel counts: {counts}"
            )

        z_scores = (arr - arr.mean()) / std
        max_idx = int(np.argmax(z_scores))

        if z_scores[max_idx] >= ZSCORE_THRESHOLD:
            exam_code += str(max_idx)
        else:
            bubble_area = bw * bh
            if arr[max_idx] > bubble_area * MIN_FILL_RATIO:
                exam_code += str(max_idx)
            else:
                raise ValueError(
                    f"Chữ số thứ {digit_idx + 1}: Z-score quá thấp ({z_scores[max_idx]:.2f}). "
                    f"Pixel counts: {counts}"
                )

    return exam_code


def _read_code_grid_divide(binary: np.ndarray,
                           num_digits: int,
                           choices_per_digit: int) -> str:
    """Fallback: chia đều ROI thành grid để đọc (phương pháp cũ)."""
    bubble_height = binary.shape[0] // choices_per_digit
    bubble_width = binary.shape[1] // num_digits

    if bubble_height == 0 or bubble_width == 0:
        raise ValueError(
            f"Kích thước bubble = 0. ROI: {binary.shape}, "
            f"digits: {num_digits}, choices: {choices_per_digit}"
        )

    exam_code = ""
    for digit_idx in range(num_digits):
        counts = []
        for choice_idx in range(choices_per_digit):
            y1 = choice_idx * bubble_height
            y2 = (choice_idx + 1) * bubble_height
            x1 = digit_idx * bubble_width
            x2 = (digit_idx + 1) * bubble_width
            bubble = binary[y1:y2, x1:x2]
            counts.append(int(cv2.countNonZero(bubble)))

        arr = np.array(counts, dtype=float)
        std = arr.std()
        if std < 10:
            raise ValueError(
                f"Chữ số thứ {digit_idx + 1}: Không phân biệt được ô nào được tô. "
                f"Pixel counts: {counts}"
            )
        z_scores = (arr - arr.mean()) / std
        max_idx = int(np.argmax(z_scores))

        if z_scores[max_idx] >= ZSCORE_THRESHOLD:
            exam_code += str(max_idx)
        else:
            bubble_area = bubble_height * bubble_width
            if arr[max_idx] > bubble_area * MIN_FILL_RATIO:
                exam_code += str(max_idx)
            else:
                raise ValueError(
                    f"Chữ số thứ {digit_idx + 1}: Z-score quá thấp ({z_scores[max_idx]:.2f}). "
                    f"Pixel counts: {counts}"
                )
    return exam_code


def _cluster_1d_simple(values: list, min_gap: float = 15) -> list:
    """Phân cụm 1D đơn giản: gộp giá trị gần nhau thành 1 cụm."""
    if not values:
        return []
    clusters = [[values[0]]]
    for v in values[1:]:
        if v - clusters[-1][-1] < min_gap:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [int(np.median(c)) for c in clusters]


def read_student_id(student_id_region: np.ndarray,
                    num_digits: int = 6,
                    threshold_method: str = "otsu") -> str:
    """
    Đọc số báo danh (SBD) từ vùng ROI.

    Dùng HoughCircles để tự phát hiện vị trí bubble,
    không phụ thuộc vào ROI cắt chính xác.

    Args:
        student_id_region: Ảnh vùng chứa SBD
        num_digits: Số chữ số (mặc định 6)
        threshold_method: Phương pháp phân ngưỡng

    Returns:
        SBD dạng string (VD: "012356")
    """
    return read_exam_code(
        student_id_region,
        num_digits=num_digits,
        choices_per_digit=10,
        threshold_method=threshold_method
    )


# ============================================================================
# HÀM TIỆN ÍCH NỘI BỘ
# ============================================================================

def _phan_nguong(gray: np.ndarray, method: str) -> np.ndarray:
    """Áp dụng phân ngưỡng theo phương pháp chỉ định."""
    if method == "adaptive":
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )
    elif method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary
    elif method == "binary":
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        return binary
    else:
        raise ValueError(
            f"Phương pháp phân ngưỡng không hợp lệ: '{method}'. "
            "Hợp lệ: 'adaptive', 'otsu', 'binary'"
        )


# ============================================================================
# HÀM HIỂN THỊ DEBUG
# ============================================================================

def visualize_all_regions(warped_image: np.ndarray,
                          rois: Dict[str, Tuple[int, int, int, int]] = None
                          ) -> np.ndarray:
    """
    Vẽ tất cả các vùng ROI lên ảnh để kiểm tra.

    Args:
        warped_image: Ảnh đã nắn chỉnh
        rois: Dict tọa độ {'sbd': (...), 'ma_de': (...), 'dap_an': (...)}
              Nếu None → tự động detect

    Returns:
        Ảnh có vẽ tất cả các khung ROI
    """
    vis = warped_image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    # Auto-detect nếu chưa có
    if rois is None:
        try:
            anchors = phat_hien_anchor(warped_image)
            rois = phan_loai_vung_roi(anchors, *warped_image.shape[:2])
        except ValueError:
            return vis

    # Vẽ từng vùng
    colors = {
        'sbd': ((255, 0, 0), "SBD"),         # Xanh dương
        'ma_de': ((0, 255, 0), "Ma De"),       # Xanh lá
        'dap_an': ((0, 0, 255), "Dap An"),     # Đỏ
    }

    for key, (color, label) in colors.items():
        if key in rois:
            x, y, w, h = rois[key]
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            cv2.putText(vis, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return vis


def visualize_anchors(warped_image: np.ndarray,
                      anchors: List[Dict] = None) -> np.ndarray:
    """
    Vẽ các anchor markers đã phát hiện lên ảnh.

    Args:
        warped_image: Ảnh đã nắn chỉnh
        anchors: Danh sách anchor. None → tự detect

    Returns:
        Ảnh có vẽ anchor markers
    """
    vis = warped_image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if anchors is None:
        try:
            anchors = phat_hien_anchor(warped_image)
        except ValueError:
            return vis

    for i, a in enumerate(anchors):
        cv2.rectangle(vis, (a['x'], a['y']),
                      (a['x'] + a['w'], a['y'] + a['h']),
                      (0, 255, 255), 2)
        cv2.putText(vis, f"#{i+1}", (a['x'], a['y'] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.circle(vis, (int(a['cx']), int(a['cy'])), 3, (0, 0, 255), -1)

    return vis
