"""
Module cấu hình cho hệ thống chấm trắc nghiệm tự động (OMR).

Module này chứa các hằng số, kích thước, ngưỡng và tham số cấu hình
cho toàn bộ pipeline xử lý.
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
WARPED_IMAGE_WIDTH = 800
WARPED_IMAGE_HEIGHT = 1200

# Tham số cho Gaussian Blur
GAUSSIAN_KERNEL_SIZE = 5  # Phải là số lẻ

# Tham số cho Median Blur
MEDIAN_KERNEL_SIZE = 5  # Phải là số lẻ

# Tham số cho Canny Edge Detection
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150


# ============================================================================
# CẤU HÌNH VÙNG ROI (Region of Interest)
# ============================================================================

# Tọa độ vùng chứa các ô đáp án (cần điều chỉnh theo template đề thi)
ROI_X = 100  # Tọa độ x góc trên-trái
ROI_Y = 200  # Tọa độ y góc trên-trái
ROI_WIDTH = 600  # Chiều rộng ROI
ROI_HEIGHT = 800  # Chiều cao ROI


# ============================================================================
# CẤU HÌNH ĐỀ THI
# ============================================================================

# Số lượng câu hỏi trong đề thi
NUM_QUESTIONS = 40

# Số lựa chọn mỗi câu (A, B, C, D)
CHOICES_PER_QUESTION = 4

# Danh sách các lựa chọn
CHOICES = ['A', 'B', 'C', 'D']


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
    # TODO: Bước 1 - Kiểm tra GAUSSIAN_KERNEL_SIZE và MEDIAN_KERNEL_SIZE là số lẻ
    # TODO: Bước 2 - Kiểm tra ADAPTIVE_BLOCK_SIZE là số lẻ
    # TODO: Bước 3 - Kiểm tra NUM_QUESTIONS > 0 và CHOICES_PER_QUESTION > 0
    # TODO: Bước 4 - Kiểm tra BUBBLE_FILLED_THRESHOLD trong khoảng [0, 1]
    # TODO: Bước 5 - Kiểm tra CANNY_LOW_THRESHOLD < CANNY_HIGH_THRESHOLD
    # TODO: Bước 6 - Return True nếu tất cả hợp lệ, False nếu không
    pass
