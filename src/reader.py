"""OMR Reader: Đọc SBD, mã đề, đáp án từ phiếu trắc nghiệm."""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict

from src.config import (
    ANCHOR_MIN_AREA, ANCHOR_MAX_AREA, ANCHOR_MIN_EXTENT,
    ANCHOR_MIN_SOLIDITY, ZSCORE_THRESHOLD, MIN_FILL_RATIO,
)


def phat_hien_anchor(warped_image: np.ndarray) -> List[Dict]:
    if warped_image.ndim == 3:
        gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = warped_image.copy()

    # Otsu threshold: tách foreground/background tự động (không cần chỉ định ngưỡng)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological close: nối contour bị đứt do nhiễu (giúp anchor liền mạch)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Tìm tất cả contour (dùng RETR_LIST để tìm cả contour bị bao bọc bên trong)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Lọc anchor theo 4 tiêu chí: area + extent + solidity + aspect_ratio
    anchors = []
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Lọc diện tích: anchor ~ 180-700 px², bubble nhỏ hơn (~180-250 px²)
        if area < ANCHOR_MIN_AREA or area > ANCHOR_MAX_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h

        # Aspect ratio: anchor phải gần vuông (w/h ≤ 1.5)
        # Loại bỏ chữ, đường kẻ dài-hẹp bị nhầm là anchor
        aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 999
        if aspect > 1.5:
            continue

        # Extent = area / bounding_rect: phân biệt hình vuông vs tròn
        # Anchor vuông: extent > 0.72 | Bubble tròn: extent < 0.70
        extent = area / rect_area if rect_area > 0 else 0
        if extent < ANCHOR_MIN_EXTENT:
            continue

        # Solidity = area / convex_hull: phân biệt hình lồi vs lõm
        # Anchor đặc: solidity > 0.93 | Bubble có lỗ: solidity < 0.9
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < ANCHOR_MIN_SOLIDITY:
            continue

        # Tính tâm chính xác bằng moments (sub-pixel) thay vì trung bình
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

    # Sắp xếp theo y rồi x (từ trên xuống, trái sang phải) để dễ định vị
    anchors.sort(key=lambda a: (a['cy'], a['cx']))

    return anchors


def phan_loai_vung_roi(anchors: List[Dict],
                       img_h: int = 1200,
                       img_w: int = 800) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Phân loại anchor thành các vùng ROI dựa trên vị trí thực tế trên phiếu.

    Cấu trúc phiếu (từ trên xuống dưới):
      - Hàng anchor trên:        2 anchor góc trên (cy ~ 8-15% h)
      - Hàng anchor giữa-trái:   1-2 anchor giữa trái/phải (cy ~ 35-45% h)
      - Hàng anchor phân cách:   3-5 anchor ngang (cy ~ 55-68% h) — ranh giới SBD|Đáp án
      - Hàng anchor dưới:        2-3 anchor góc dưới (cy ~ 85-95% h)

    Strategy: cluster anchor theo trục y → tìm hàng gần 60% h nhất làm
    ranh giới SBD/MĐề vs Đáp án.
    """
    # ── Bước 1: Cluster anchor theo trục y (tìm các "hàng anchor") ──
    sorted_by_y = sorted(anchors, key=lambda a: a['cy'])

    # Khoảng cách tối thiểu giữa 2 hàng = 5% chiều cao ảnh
    row_gap = img_h * 0.05
    rows: List[List[Dict]] = []
    current_row = [sorted_by_y[0]]
    for a in sorted_by_y[1:]:
        if a['cy'] - current_row[-1]['cy'] < row_gap:
            current_row.append(a)
        else:
            rows.append(current_row)
            current_row = [a]
    rows.append(current_row)

    # ── Bước 2: Tìm hàng anchor phân cách SBD/MĐề ↔ Đáp án ──
    # Hàng phân cách nằm trong vùng 50~70% chiều cao
    sep_zone_min = img_h * 0.50
    sep_zone_max = img_h * 0.72

    sep_row = None
    for row in rows:
        row_y = float(np.mean([a['cy'] for a in row]))
        if sep_zone_min <= row_y <= sep_zone_max:
            # Ưu tiên hàng có nhiều anchor nhất
            if sep_row is None or len(row) > len(sep_row):
                sep_row = row

    # Fallback: lấy hàng gần 62% nhất
    if sep_row is None:
        target_y = img_h * 0.62
        sep_row = min(rows, key=lambda r: abs(float(np.mean([a['cy'] for a in r])) - target_y))

    sep_y = float(np.mean([a['cy'] for a in sep_row]))

    # ── Bước 3: Phân anchor thành nhóm trên và nhóm dưới ──
    nhom_tren = [a for a in anchors if a['cy'] < sep_y - 5]
    nhom_duoi = [a for a in anchors if a['cy'] >= sep_y - 5]

    roi_sbd = None
    roi_ma_de = None

    # ── Bước 4: Tính ROI SBD và Mã Đề từ nhóm trên ──
    if nhom_tren:
        nhom_tren_sorted = sorted(nhom_tren, key=lambda a: a['cx'])

        # Anchor phân cách SBD | Mã đề: nằm quanh 45% chiều rộng
        mid_x = img_w * 0.45
        phan_cach = [a for a in nhom_tren_sorted
                     if abs(a['cx'] - mid_x) < img_w * 0.18]

        # Y start: lấy anchor cao nhất (góc trên) + offset nhỏ
        top_row_anchors = [a for a in nhom_tren if a['cy'] < img_h * 0.40]
        if top_row_anchors:
            y_start = int(max(a['cy'] + a['h'] for a in top_row_anchors))
        else:
            y_start = int(img_h * 0.30)

        # Y end: ngay phía trên hàng anchor phân cách
        anchor_h_avg = float(np.mean([a['h'] for a in sep_row]))
        y_end = int(sep_y - anchor_h_avg * 0.5)
        y_end = min(y_end, img_h - 1)

        if phan_cach:
            sep = phan_cach[0]

            # SBD: luôn lấy từ 10% chiều rộng để bao trọn tất cả các cột SBD
            x_start_sbd = int(img_w * 0.10)
            x_end_sbd = int(sep['cx'] - sep['w'] * 0.5)
            roi_sbd = (x_start_sbd, y_start,
                       max(1, x_end_sbd - x_start_sbd), max(1, y_end - y_start))

            # Mã đề: từ anchor phân cách đến bên phải
            # Giữ đủ rộng (75%) cho cả phiếu có/không có text 'FILLING ID...'
            # Lọc cột text dọc được xử lý trong read_exam_code bằng z-score
            x_start_md = int(sep['cx'] + sep['w'] * 0.5)
            x_end_md = int(img_w * 0.75)
            # Đảm bảo đủ rộng tối thiểu 18% để chứa 3 cột bubble
            x_end_md = max(x_end_md, x_start_md + int(img_w * 0.18))
            x_end_md = min(x_end_md, img_w - 1)
            roi_ma_de = (x_start_md, y_start,
                         max(1, x_end_md - x_start_md), max(1, y_end - y_start))
        else:
            # Không tìm được anchor phân cách → chia đôi theo chiều rộng
            x_mid = img_w // 2
            roi_sbd = (int(img_w * 0.10), y_start,
                       max(1, int(x_mid * 0.9) - int(img_w * 0.10)), max(1, y_end - y_start))
            roi_ma_de = (x_mid, y_start,
                         max(1, int(img_w * 0.75) - x_mid), max(1, y_end - y_start))

    # ── Bước 5: Tính ROI Đáp án từ nhóm dưới ──
    roi_dap_an = None

    if nhom_duoi:
        anchor_h_avg = float(np.mean([a['h'] for a in sep_row]))

        # Y top: ngay dưới hàng anchor phân cách (offset nhỏ để không cắt hàng đầu)
        y_top = int(sep_y + anchor_h_avg * 1.0)

        # Tìm hàng anchor dưới cùng (cy > 80% h) để làm ranh giới dưới
        bottom_anchors = [a for a in nhom_duoi if a['cy'] > img_h * 0.80]
        if bottom_anchors:
            y_bot = int(max(a['cy'] - a['h'] * 0.5 for a in bottom_anchors))
        else:
            y_bot = int(img_h * 0.98)

        # X: dùng x min/max từ TẤT CẢ anchor của nhóm dưới (bao gồm sep_row)
        # để đảm bảo bao trọn toàn bộ 2 cột đáp án (câu 1-10 và câu 11-20)
        all_x_below = [a['cx'] for a in nhom_duoi]
        all_x_sep = [a['cx'] for a in sep_row]
        all_x_combined = all_x_below + all_x_sep
        if len(all_x_combined) >= 2:
            x_min = int(min(all_x_combined) - 30)
            x_max = int(max(all_x_combined) + 30)
        else:
            x_min = int(min(all_x_combined) - 30)
            x_max = int(max(all_x_combined) + 30)

        # Clip vào giới hạn ảnh
        x_min = max(0, x_min)
        x_max = min(img_w, x_max)
        y_top = max(0, y_top)
        y_bot = min(img_h, y_bot)

        roi_dap_an = (x_min, y_top, max(1, x_max - x_min), max(1, y_bot - y_top))

    # ── Fallback: dùng tọa độ cứng nếu auto-detect thất bại ──
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


def extract_exam_code_region(warped_image: np.ndarray,
                             roi_x: int = None,
                             roi_y: int = None,
                             roi_width: int = None,
                             roi_height: int = None) -> np.ndarray:
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
    if roi_x is None or roi_y is None or roi_width is None or roi_height is None:
        anchors = phat_hien_anchor(warped_image)
        rois = phan_loai_vung_roi(anchors, *warped_image.shape[:2])
        roi_x, roi_y, roi_width, roi_height = rois['dap_an']

    return _cat_vung_roi(warped_image, roi_x, roi_y, roi_width, roi_height)


def _cat_vung_roi(image: np.ndarray,
                  x: int, y: int, w: int, h: int) -> np.ndarray:
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


def read_exam_code(exam_code_region: np.ndarray,
                   num_digits: int = 3,
                   choices_per_digit: int = 10,
                   threshold_method: str = "otsu") -> str:
    # Tiền xử lý: chuyển xám + blur
    if exam_code_region.ndim == 3:
        gray = cv2.cvtColor(exam_code_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = exam_code_region.copy()

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Phân ngưỡng để đọc pixel tô đậm
    binary = _phan_nguong(gray, threshold_method)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # HoughCircles phát hiện bubble
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=12,
        param1=50, param2=20,
        minRadius=6, maxRadius=16
    )

    if circles is None or len(circles[0]) < num_digits * choices_per_digit * 0.5:
        # Fallback: dùng phương pháp chia đều cũ nếu HoughCircles thất bại
        return _read_code_grid_divide(binary, num_digits, choices_per_digit)

    circles = np.round(circles[0]).astype(int)

    # Phân cụm x → cột (mỗi cột = 1 chữ số)
    # Dùng thuật toán Peak Finding (Histogram) thay vì gộp tuần tự để tránh bị ảnh hưởng bởi nhiễu (outliers)
    all_cx = sorted([int(c[0]) for c in circles])
    
    # 1. Tạo histogram với bin_size = 5px
    max_x = max(all_cx) if all_cx else 0
    hist, bins = np.histogram(all_cx, bins=np.arange(0, max_x + 10, 5))
    
    # 2. Tìm các đỉnh (peaks) cục bộ
    peaks = []
    for i in range(1, len(hist)-1):
        # Đỉnh phải cao hơn xung quanh và có ít nhất 2 vòng tròn
        if hist[i] > hist[i-1] and hist[i] >= hist[i+1] and hist[i] > 1:
            peaks.append(bins[i] + 2.5)
            
    # 3. Gộp các đỉnh quá gần nhau (do 1 bong bóng bị tách đôi bởi số in bên trong)
    # Khoảng cách giữa 2 cột gần nhất (phiếu 04) là ~22px.
    # Khoảng cách giữa 2 nửa bong bóng (phiếu 01-03) là ~12-15px.
    # Dùng ngưỡng 16px để tách biệt hoàn hảo 2 trường hợp.
    col_centers = []
    for p in peaks:
        if not col_centers or p - col_centers[-1] >= 16:
            col_centers.append(p)
        else:
            col_centers[-1] = (col_centers[-1] + p) / 2
            
    col_centers = [int(c) for c in col_centers]

    # Phân cụm y → hàng (0-9)
    all_cy = sorted([int(c[1]) for c in circles])
    
    max_y = max(all_cy) if all_cy else 0
    hist_y, bins_y = np.histogram(all_cy, bins=np.arange(0, max_y + 10, 5))
    
    peaks_y = []
    for i in range(1, len(hist_y)-1):
        if hist_y[i] > hist_y[i-1] and hist_y[i] >= hist_y[i+1] and hist_y[i] > 0:
            peaks_y.append(bins_y[i] + 2.5)
            
    row_centers = []
    for p in peaks_y:
        if not row_centers or p - row_centers[-1] >= 16:
            row_centers.append(p)
        else:
            row_centers[-1] = (row_centers[-1] + p) / 2
            
    row_centers = [int(r) for r in row_centers]

    # Lọc: chỉ giữ cột có đủ bubble (≥ 50% số hàng)
    col_counts = []
    for cc in col_centers:
        count = sum(1 for c in circles if abs(int(c[0]) - cc) < 14)
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
    print(f"DEBUG: col_counts={col_counts}, valid_cols={valid_cols}, digit_cols={digit_cols}")

    # Chỉ giữ đúng số hàng cần thiết (10)
    # Lấy N hàng CUỐI — vì hàng header nằm ở trên cùng
    # Validate: chọn ra đúng `choices_per_digit` hàng có khoảng cách đều nhau nhất
    if len(row_centers) > choices_per_digit and len(row_centers) >= 3:
        gaps = [row_centers[i+1] - row_centers[i] for i in range(len(row_centers)-1)]
        median_gap = sorted(gaps)[len(gaps)//2]
        
        best_start = 0
        min_err = float('inf')
        for i in range(len(row_centers) - choices_per_digit + 1):
            err = 0
            for j in range(choices_per_digit - 1):
                err += abs((row_centers[i+j+1] - row_centers[i+j]) - median_gap)
            if err < min_err:
                min_err = err
                best_start = i
        row_centers = row_centers[best_start : best_start + choices_per_digit]
    elif len(row_centers) > choices_per_digit:
        row_centers = row_centers[-choices_per_digit:]
        
    print(f"DEBUG: row_centers={row_centers}")

    if len(row_centers) < choices_per_digit:
        return _read_code_grid_divide(binary, num_digits, choices_per_digit, digit_cols)

    # Khử nhiễu nét chữ in (số bên trong bubble) bằng erosion mạnh hơn trên toàn bộ ảnh
    # Việc thực hiện trước vòng lặp giúp tránh lỗi viền (border effects) khi cắt nhỏ ảnh
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_eroded = cv2.erode(binary, kernel_erode, iterations=1)

    # Đọc từng chữ số bằng z-score
    avg_r = int(np.mean([int(c[2]) for c in circles]))
    # Dùng bounding box nhỏ hơn một chút so với bong bóng để tập trung đếm pixel lõi,
    # tránh lấn sang phần text hướng dẫn bên cạnh (gây nhiễu cho cột cuối).
    bw = int(avg_r * 1.6)
    bh = int(avg_r * 1.6)

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
            region = binary_eroded[y1:y2, x1:x2]
            counts.append(int(cv2.countNonZero(region)))

        # Z-score: tìm chữ số nổi bật nhất (pixel tô nhiều nhất)
        arr = np.array(counts, dtype=float)
        std = arr.std()

        if std < 10:
            raise ValueError(
                f"Chữ số thứ {digit_idx + 1}: Không phân biệt được ô nào được tô. "
                f"Pixel counts: {counts}"
            )

        z_scores = (arr - arr.mean()) / std
        max_idx = int(np.argmax(z_scores))
        print(f"Digit {digit_idx}: z_scores={z_scores}, counts={counts}")

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
                           choices_per_digit: int,
                           digit_cols: list = None) -> str:
    """Fallback: chia đều ROI thành grid để đọc (phương pháp cũ)."""
    bubble_height = binary.shape[0] // choices_per_digit
    bubble_width = binary.shape[1] // num_digits

    if bubble_height == 0 or bubble_width == 0:
        raise ValueError(
            f"Kích thước bubble = 0. ROI: {binary.shape}, "
            f"digits: {num_digits}, choices: {choices_per_digit}"
        )

    # Khử nhiễu nét chữ in bằng erosion trên toàn bộ ảnh
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_eroded = cv2.erode(binary, kernel_erode, iterations=1)

    exam_code = ""
    for digit_idx in range(num_digits):
        counts = []
        for choice_idx in range(choices_per_digit):
            y1 = choice_idx * bubble_height
            y2 = (choice_idx + 1) * bubble_height
            if digit_cols and len(digit_cols) == num_digits:
                col_x = digit_cols[digit_idx]
                bw = min(bubble_height, 24)  # Giới hạn chiều rộng bubble
                x1 = max(0, col_x - bw // 2)
                x2 = min(binary.shape[1], col_x + bw // 2)
            else:
                x1 = digit_idx * bubble_width
                x2 = (digit_idx + 1) * bubble_width
                
            bubble = binary_eroded[y1:y2, x1:x2]
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
    return read_exam_code(
        student_id_region,
        num_digits=num_digits,
        choices_per_digit=10,
        threshold_method=threshold_method
    )


def _phan_nguong(gray: np.ndarray, method: str) -> np.ndarray:
    if method == "adaptive":
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            51, 2  # Tăng blockSize lên 51 để bong bóng đen không bị làm rỗng
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


def visualize_all_regions(warped_image: np.ndarray,
                          rois: Dict[str, Tuple[int, int, int, int]] = None
                          ) -> np.ndarray:
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
