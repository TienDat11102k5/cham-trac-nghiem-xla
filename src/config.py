"""
Module cấu hình cho hệ thống chấm trắc nghiệm tự động (OMR).

Module này chứa các hằng số, kích thước, ngưỡng và tham số cấu hình
cho toàn bộ pipeline xử lý.

LƯU Ý VỀ ĐỒNG BỘ THAM SỐ:
- File này lưu các giá trị THAM KHẢO và giá trị đang dùng THỰC TẾ trong main.py
- Một số module có default khác để linh hoạt khi test
- Xem chi tiết trong docs/THAM_SO_CAU_HINH.md
"""

from typing import Dict, Tuple


# ============================================================================
# CẤU HÌNH ĐƯỜNG DẪN THƯ MỤC
# ============================================================================

DATA_DIR = "data"
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
OUTPUT_DIR = "output"
NOTEBOOKS_DIR = "notebooks"


# ============================================================================
# CẤU HÌNH XỬ LÝ ẢNH
# ============================================================================

# Kích thước ảnh sau khi nắn chỉnh (pixels)
# Giá trị này khớp với main.py: nan_chinh_anh(anh_goc, cac_goc, chieu_rong=800, chieu_cao=1200)
WARPED_IMAGE_WIDTH = 800
WARPED_IMAGE_HEIGHT = 1200

# Tham số cho Gaussian Blur
# Giá trị mặc định trong preprocessing.py: kich_thuoc=5
# Giá trị dùng trong main.py: kich_thuoc=5
GAUSSIAN_KERNEL_SIZE = 5  # Phải là số lẻ

# Tham số cho Median Blur
# Giá trị mặc định trong preprocessing.py: kich_thuoc=5
MEDIAN_KERNEL_SIZE = 5  # Phải là số lẻ

# Tham số cho Canny Edge Detection
# Giá trị mặc định trong transform.py: nguong_thap=30, nguong_cao=100
# Giá trị dùng trong main.py: nguong_thap=50, nguong_cao=150
CANNY_LOW_THRESHOLD = 50   # Giá trị thực tế đang dùng trong main.py
CANNY_HIGH_THRESHOLD = 150  # Giá trị thực tế đang dùng trong main.py

# LƯU Ý: 
# - preprocessing.py có default khác (kich_thuoc=5) - KHỚP với config
# - transform.py có default khác (nguong_thap=30, nguong_cao=100) - KHÁC với config
# - main.py ghi đè bằng giá trị cụ thể (50, 150) - KHỚP với config này


# ============================================================================
# CẤU HÌNH PHÁT HIỆN ANCHOR MARKERS (Tự động tìm vùng ROI)
# ============================================================================
# Anchor markers = các ô vuông đen in sẵn trên phiếu trắc nghiệm.
# Hệ thống dùng anchor để tính tọa độ ROI tương đối → chạy đúng mọi ảnh.
# Giá trị dưới đây xác định bằng phân tích thực tế trên ảnh 800×1200.

# Diện tích anchor marker (pixel²) — lọc contour theo diện tích
ANCHOR_MIN_AREA = 250       # Loại bỏ nhiễu nhỏ
ANCHOR_MAX_AREA = 600       # Loại bỏ bubble và vùng lớn

# Extent tối thiểu — tỉ lệ area/bounding_rect
# Anchor (vuông đặc): extent > 0.82 | Bubble (tròn): extent < 0.76
ANCHOR_MIN_EXTENT = 0.82

# Solidity tối thiểu — tỉ lệ area/convex_hull_area
ANCHOR_MIN_SOLIDITY = 0.93

# Z-score threshold cho đọc bubble (reader + grader)
# Z >= threshold → xác nhận bubble được tô
# Z-score miễn nhiễm tẩy xóa bẩn: bubble bị tẩy có z thấp hơn bubble mới tô
ZSCORE_THRESHOLD = 1.4

# Tỉ lệ pixel tối thiểu so với diện tích bubble (fallback khi std ~ 0)
MIN_FILL_RATIO = 0.15

# ============================================================================
# CẤU HÌNH VÙNG ROI (Region of Interest) — CHỈ DÙNG LÀM THAM KHẢO
# ============================================================================
# ⚠️  DEPRECATED: Các giá trị dưới đây chỉ để tham khảo.
# Hệ thống hiện dùng auto-detect qua anchor markers (xem reader.py).
# Giá trị gốc đo từ test_sheet_02.jpg (1440x2560) → chuyển về 800x1200.

ROI_X = 157        # Tọa độ x góc trên-trái vùng đáp án
ROI_Y = 797        # Tọa độ y góc trên-trái vùng đáp án
ROI_WIDTH = 391    # Chiều rộng ROI
ROI_HEIGHT = 319   # Chiều cao ROI

EXAM_CODE_ROI_X = 406
EXAM_CODE_ROI_Y = 437
EXAM_CODE_ROI_WIDTH = 112
EXAM_CODE_ROI_HEIGHT = 322

STUDENT_ID_ROI_X = 161
STUDENT_ID_ROI_Y = 439
STUDENT_ID_ROI_WIDTH = 207
STUDENT_ID_ROI_HEIGHT = 323


# ============================================================================
# CẤU HÌNH ĐỀ THI
# ============================================================================

# Số lượng câu hỏi trong đề thi
NUM_QUESTIONS = 40

# Số lựa chọn mỗi câu (A, B, C, D)
CHOICES_PER_QUESTION = 4

# Danh sách các lựa chọn
CHOICES = ['A', 'B', 'C', 'D']

# Cấu hình mã đề thi
NUM_EXAM_CODE_DIGITS = 3  # Số chữ số của mã đề (101, 102, 103)
CHOICES_PER_EXAM_CODE_DIGIT = 10  # Mỗi chữ số có 10 lựa chọn (0-9)

# Cấu hình mã sinh viên (tùy chọn)
NUM_STUDENT_ID_DIGITS = 8  # Số chữ số của mã sinh viên
CHOICES_PER_STUDENT_ID_DIGIT = 10  # Mỗi chữ số có 10 lựa chọn (0-9)


# ============================================================================
# CẤU HÌNH PHÂN NGƯỠNG (THRESHOLDING)
# ============================================================================

# Phương pháp phân ngưỡng mặc định
THRESHOLD_METHOD = "adaptive"  # Các giá trị: "adaptive", "otsu", "binary"

# Tham số cho Adaptive Threshold
ADAPTIVE_BLOCK_SIZE = 11  # Phải là số lẻ
ADAPTIVE_C = 2

# Tham số cho Binary Threshold
BINARY_THRESHOLD_VALUE = 127


# ============================================================================
# CẤU HÌNH CHẤM ĐIỂM
# ============================================================================

# Ngưỡng để xác định ô có được tô hay không (% diện tích)
BUBBLE_FILLED_THRESHOLD = 0.5  # 50% diện tích bubble

# Thang điểm
MAX_SCORE = 10.0


# ============================================================================
# ĐÁP ÁN MẪU (ANSWER KEY)
# ============================================================================

# Đáp án chuẩn cho đề thi mẫu (40 câu)
SAMPLE_ANSWER_KEY: Dict[int, str] = {
    1: 'A', 2: 'C', 3: 'B', 4: 'D', 5: 'A',
    6: 'B', 7: 'C', 8: 'D', 9: 'A', 10: 'B',
    11: 'C', 12: 'D', 13: 'A', 14: 'B', 15: 'C',
    16: 'D', 17: 'A', 18: 'B', 19: 'C', 20: 'D',
    21: 'A', 22: 'B', 23: 'C', 24: 'D', 25: 'A',
    26: 'B', 27: 'C', 28: 'D', 29: 'A', 30: 'B',
    31: 'C', 32: 'D', 33: 'A', 34: 'B', 35: 'C',
    36: 'D', 37: 'A', 38: 'B', 39: 'C', 40: 'D',
}


# ============================================================================
# CẤU HÌNH HIỂN THỊ VÀ DEBUG
# ============================================================================

# Có hiển thị ảnh trung gian trong quá trình xử lý không
SHOW_INTERMEDIATE_IMAGES = False

# Có lưu ảnh trung gian vào thư mục output không
SAVE_INTERMEDIATE_IMAGES = True

# Độ trễ khi hiển thị ảnh (milliseconds, 0 = chờ phím bất kỳ)
DISPLAY_DELAY = 0


# ============================================================================
# HÀM TIỆN ÍCH
# ============================================================================

def get_roi_coordinates() -> Tuple[int, int, int, int]:
    """
    Trả về tọa độ ROI dưới dạng tuple.
    
    Returns:
        Tuple[int, int, int, int]: (x, y, width, height)
    """
    return (ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT)


def get_warped_image_size() -> Tuple[int, int]:
    """
    Trả về kích thước ảnh sau khi nắn chỉnh.
    
    Returns:
        Tuple[int, int]: (width, height)
    """
    return (WARPED_IMAGE_WIDTH, WARPED_IMAGE_HEIGHT)


def validate_config() -> bool:
    """
    Kiểm tra tính hợp lệ của các tham số cấu hình.
    
    Returns:
        bool: True nếu cấu hình hợp lệ, False nếu không.
    
    Notes:
        - Kernel size phải là số lẻ và > 0
        - Ngưỡng phải nằm trong khoảng hợp lệ
        - Số câu hỏi và lựa chọn phải > 0
    """
    errors = []

    # Bước 1 - Kiểm tra kernel size là số lẻ dương
    for name, val in [("GAUSSIAN_KERNEL_SIZE", GAUSSIAN_KERNEL_SIZE),
                      ("MEDIAN_KERNEL_SIZE", MEDIAN_KERNEL_SIZE)]:
        if val <= 0 or val % 2 == 0:
            errors.append(f"{name} phải là số lẻ dương, nhận được: {val}")

    # Bước 2 - Kiểm tra ADAPTIVE_BLOCK_SIZE là số lẻ
    if ADAPTIVE_BLOCK_SIZE <= 0 or ADAPTIVE_BLOCK_SIZE % 2 == 0:
        errors.append(f"ADAPTIVE_BLOCK_SIZE phải là số lẻ dương, nhận được: {ADAPTIVE_BLOCK_SIZE}")

    # Bước 3 - Kiểm tra số câu hỏi và lựa chọn > 0
    if NUM_QUESTIONS <= 0:
        errors.append(f"NUM_QUESTIONS phải > 0, nhận được: {NUM_QUESTIONS}")
    if CHOICES_PER_QUESTION <= 0:
        errors.append(f"CHOICES_PER_QUESTION phải > 0, nhận được: {CHOICES_PER_QUESTION}")

    # Bước 4 - Kiểm tra BUBBLE_FILLED_THRESHOLD trong [0, 1]
    if not (0.0 <= BUBBLE_FILLED_THRESHOLD <= 1.0):
        errors.append(f"BUBBLE_FILLED_THRESHOLD phải trong [0, 1], nhận được: {BUBBLE_FILLED_THRESHOLD}")

    # Bước 5 - Kiểm tra Canny thresholds
    if CANNY_LOW_THRESHOLD >= CANNY_HIGH_THRESHOLD:
        errors.append(
            f"CANNY_LOW_THRESHOLD ({CANNY_LOW_THRESHOLD}) "
            f"phải < CANNY_HIGH_THRESHOLD ({CANNY_HIGH_THRESHOLD})"
        )

    # Bước 6 - Kiểm tra anchor params
    if ANCHOR_MIN_AREA >= ANCHOR_MAX_AREA:
        errors.append(
            f"ANCHOR_MIN_AREA ({ANCHOR_MIN_AREA}) "
            f"phải < ANCHOR_MAX_AREA ({ANCHOR_MAX_AREA})"
        )
    if not (0.0 < ZSCORE_THRESHOLD < 10.0):
        errors.append(f"ZSCORE_THRESHOLD phải trong (0, 10), nhận được: {ZSCORE_THRESHOLD}")

    if errors:
        for e in errors:
            print(f"⚠️  Config error: {e}")
        return False

    return True

