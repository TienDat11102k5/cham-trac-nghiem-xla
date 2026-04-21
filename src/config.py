from typing import Dict, Tuple


DATA_DIR = "data"
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
OUTPUT_DIR = "output"
NOTEBOOKS_DIR = "notebooks"

WARPED_IMAGE_WIDTH = 800  # Chiều rộng ảnh sau nắn chỉnh
WARPED_IMAGE_HEIGHT = 1200  # Chiều cao ảnh sau nắn chỉnh

GAUSSIAN_KERNEL_SIZE = 5  # Kích thước kernel Gaussian blur (phải lẻ)
MEDIAN_KERNEL_SIZE = 5  # Kích thước kernel Median blur (phải lẻ)

CANNY_LOW_THRESHOLD = 50  # Ngưỡng thấp Canny edge detection
CANNY_HIGH_THRESHOLD = 150  # Ngưỡng cao Canny edge detection

ANCHOR_MIN_AREA = 250  # Diện tích tối thiểu anchor (pixel²)
ANCHOR_MAX_AREA = 600  # Diện tích tối đa anchor (pixel²)
ANCHOR_MIN_EXTENT = 0.82  # Tỉ lệ area/bounding_rect tối thiểu (phân biệt vuông vs tròn)
ANCHOR_MIN_SOLIDITY = 0.93  # Tỉ lệ area/convex_hull tối thiểu (phân biệt lồi vs lõm)

ZSCORE_THRESHOLD = 1.4  # Z-score tối thiểu để xác nhận bubble được tô
MIN_FILL_RATIO = 0.15  # Tỉ lệ pixel tối thiểu so với diện tích bubble

ROI_X = 157  # Tọa độ x vùng đáp án (tham khảo)
ROI_Y = 797  # Tọa độ y vùng đáp án (tham khảo)
ROI_WIDTH = 391  # Chiều rộng vùng đáp án (tham khảo)
ROI_HEIGHT = 319  # Chiều cao vùng đáp án (tham khảo)

EXAM_CODE_ROI_X = 406  # Tọa độ x vùng mã đề (tham khảo)
EXAM_CODE_ROI_Y = 437  # Tọa độ y vùng mã đề (tham khảo)
EXAM_CODE_ROI_WIDTH = 112  # Chiều rộng vùng mã đề (tham khảo)
EXAM_CODE_ROI_HEIGHT = 322  # Chiều cao vùng mã đề (tham khảo)

STUDENT_ID_ROI_X = 161  # Tọa độ x vùng SBD (tham khảo)
STUDENT_ID_ROI_Y = 439  # Tọa độ y vùng SBD (tham khảo)
STUDENT_ID_ROI_WIDTH = 207  # Chiều rộng vùng SBD (tham khảo)
STUDENT_ID_ROI_HEIGHT = 323  # Chiều cao vùng SBD (tham khảo)

NUM_QUESTIONS = 40  # Tổng số câu hỏi
CHOICES_PER_QUESTION = 4  # Số lựa chọn mỗi câu (A, B, C, D)
CHOICES = ['A', 'B', 'C', 'D']  # Danh sách lựa chọn

NUM_EXAM_CODE_DIGITS = 3  # Số chữ số mã đề
CHOICES_PER_EXAM_CODE_DIGIT = 10  # Mỗi chữ số có 10 lựa chọn (0-9)

NUM_STUDENT_ID_DIGITS = 8  # Số chữ số SBD
CHOICES_PER_STUDENT_ID_DIGIT = 10  # Mỗi chữ số có 10 lựa chọn (0-9)

THRESHOLD_METHOD = "adaptive"  # Phương pháp phân ngưỡng: "adaptive", "otsu", "binary"
ADAPTIVE_BLOCK_SIZE = 11  # Kích thước block adaptive threshold (phải lẻ)
ADAPTIVE_C = 2  # Hằng số C cho adaptive threshold
BINARY_THRESHOLD_VALUE = 127  # Ngưỡng cho binary threshold

BUBBLE_FILLED_THRESHOLD = 0.5  # Tỉ lệ pixel tối thiểu để xác nhận bubble tô (50%)
MAX_SCORE = 10.0  # Thang điểm tối đa

SHOW_INTERMEDIATE_IMAGES = False  # Có hiển thị ảnh trung gian không
SAVE_INTERMEDIATE_IMAGES = True  # Có lưu ảnh trung gian không
DISPLAY_DELAY = 0  # Độ trễ hiển thị ảnh (ms, 0 = chờ phím)


def get_roi_coordinates() -> Tuple[int, int, int, int]:
    return (ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT)


def get_warped_image_size() -> Tuple[int, int]:
    return (WARPED_IMAGE_WIDTH, WARPED_IMAGE_HEIGHT)


def validate_config() -> bool:
    errors = []

    for name, val in [("GAUSSIAN_KERNEL_SIZE", GAUSSIAN_KERNEL_SIZE),
                      ("MEDIAN_KERNEL_SIZE", MEDIAN_KERNEL_SIZE)]:
        if val <= 0 or val % 2 == 0:
            errors.append(f"{name} phải là số lẻ dương, nhận được: {val}")

    if ADAPTIVE_BLOCK_SIZE <= 0 or ADAPTIVE_BLOCK_SIZE % 2 == 0:
        errors.append(f"ADAPTIVE_BLOCK_SIZE phải là số lẻ dương, nhận được: {ADAPTIVE_BLOCK_SIZE}")

    if NUM_QUESTIONS <= 0:
        errors.append(f"NUM_QUESTIONS phải > 0, nhận được: {NUM_QUESTIONS}")
    if CHOICES_PER_QUESTION <= 0:
        errors.append(f"CHOICES_PER_QUESTION phải > 0, nhận được: {CHOICES_PER_QUESTION}")

    if not (0.0 <= BUBBLE_FILLED_THRESHOLD <= 1.0):
        errors.append(f"BUBBLE_FILLED_THRESHOLD phải trong [0, 1], nhận được: {BUBBLE_FILLED_THRESHOLD}")

    if CANNY_LOW_THRESHOLD >= CANNY_HIGH_THRESHOLD:
        errors.append(
            f"CANNY_LOW_THRESHOLD ({CANNY_LOW_THRESHOLD}) "
            f"phải < CANNY_HIGH_THRESHOLD ({CANNY_HIGH_THRESHOLD})"
        )

    if ANCHOR_MIN_AREA >= ANCHOR_MAX_AREA:
        errors.append(
            f"ANCHOR_MIN_AREA ({ANCHOR_MIN_AREA}) "
            f"phải < ANCHOR_MAX_AREA ({ANCHOR_MAX_AREA})"
        )
    if not (0.0 < ZSCORE_THRESHOLD < 10.0):
        errors.append(f"ZSCORE_THRESHOLD phải trong (0, 10), nhận được: {ZSCORE_THRESHOLD}")

    if errors:
        for e in errors:
            print(f"Config error: {e}")
        return False

    return True