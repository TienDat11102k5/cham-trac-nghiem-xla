"""
Package chứa các module xử lý cho hệ thống chấm trắc nghiệm tự động (OMR).

Modules:
    - preprocessing: Tiền xử lý ảnh (đọc, chuyển xám, khử nhiễu)
    - transform: Biến đổi hình học (phát hiện biên, nắn chỉnh)
    - grader: Chấm điểm (trích xuất ROI, phân đoạn, tính điểm)
    - config: Cấu hình hệ thống
    - utils: Các hàm tiện ích
"""

__version__ = "1.0.0"
__author__ = "Computer Vision Team"

# Import các module chính để dễ dàng truy cập
from .preprocessing import doc_anh, chuyen_xam, loc_nhieu
from .transform import tim_canh, tim_goc_giay, nan_chinh_anh
from .grader import extract_bubble_grid, segment_bubbles, calculate_score

__all__ = [
    "doc_anh",
    "chuyen_xam",
    "loc_nhieu",
    "tim_canh",
    "tim_goc_giay",
    "nan_chinh_anh",
    "extract_bubble_grid",
    "segment_bubbles",
    "calculate_score",
]
