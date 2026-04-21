__version__ = "1.0.0"
__author__ = "Computer Vision Team"

from .preprocessing import doc_anh, chuyen_xam, loc_nhieu
from .transform import tim_canh, tim_goc_giay, nan_chinh_anh
from .reader import (
    phat_hien_anchor, phan_loai_vung_roi,
    extract_exam_code_region, extract_student_id_region, extract_answer_region,
    read_exam_code, read_student_id,
    visualize_all_regions, visualize_anchors,
)
from .grader import (
    extract_bubble_grid, segment_bubbles, calculate_score,
    phat_hien_luoi_bubble, grade_from_image,
)
from .config import (
    WARPED_IMAGE_WIDTH, WARPED_IMAGE_HEIGHT,
    ZSCORE_THRESHOLD, MIN_FILL_RATIO,
    NUM_QUESTIONS, CHOICES_PER_QUESTION,
)

__all__ = [
    "doc_anh",
    "chuyen_xam",
    "loc_nhieu",
    "tim_canh",
    "tim_goc_giay",
    "nan_chinh_anh",
    "phat_hien_anchor",
    "phan_loai_vung_roi",
    "extract_exam_code_region",
    "extract_student_id_region",
    "extract_answer_region",
    "read_exam_code",
    "read_student_id",
    "visualize_all_regions",
    "visualize_anchors",
    "extract_bubble_grid",
    "segment_bubbles",
    "calculate_score",
    "phat_hien_luoi_bubble",
    "grade_from_image",
    "WARPED_IMAGE_WIDTH",
    "WARPED_IMAGE_HEIGHT",
    "ZSCORE_THRESHOLD",
    "MIN_FILL_RATIO",
    "NUM_QUESTIONS",
    "CHOICES_PER_QUESTION",
]
