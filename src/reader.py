"""
Module đọc thông tin từ phiếu trắc nghiệm (OMR Reader).

Module này chứa các hàm để đọc:
- Mã đề thi (exam code)
- Mã sinh viên (student ID) - tùy chọn
- Các thông tin khác được tô đen trên phiếu

Thành viên phụ trách: [Tên thành viên]
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List


# ============================================================================
# ĐỌC MÃ ĐỀ THI (EXAM CODE)
# ============================================================================

def extract_exam_code_region(warped_image: np.ndarray,
                             roi_x: int = 406,
                             roi_y: int = 437,
                             roi_width: int = 112,
                             roi_height: int = 322) -> np.ndarray:
    """
    Cắt vùng chứa mã đề thi (thường ở góc trên bên phải).
    
    Args:
        warped_image (np.ndarray): Ảnh đã nắn chỉnh, shape (height, width) hoặc (height, width, 3)
        roi_x (int): Tọa độ x góc trên-trái của vùng mã đề. Mặc định 600.
        roi_y (int): Tọa độ y góc trên-trái của vùng mã đề. Mặc định 50.
        roi_width (int): Chiều rộng vùng mã đề. Mặc định 150.
        roi_height (int): Chiều cao vùng mã đề. Mặc định 100.
    
    Returns:
        np.ndarray: Ảnh vùng ROI chứa mã đề, shape (roi_height, roi_width)
    
    Raises:
        ValueError: Nếu ROI vượt quá giới hạn ảnh
    
    Notes:
        - Mã đề thường có 3-4 chữ số (101, 102, 103, 104)
        - Mỗi chữ số có 10 ô (0-9) xếp dọc
        - Vị trí ROI cần điều chỉnh theo template đề thi cụ thể
        - Có thể dùng công cụ visualization để xác định tọa độ chính xác
    
    Examples:
        >>> exam_code_region = extract_exam_code_region(warped_image, roi_x=600, roi_y=50)
        >>> print(exam_code_region.shape)
        (100, 150)
    """
    # Bước 1 - Kiểm tra ROI có nằm trong giới hạn ảnh không
    h, w = warped_image.shape[:2]
    
    if roi_x < 0 or roi_y < 0:
        raise ValueError(
            f"Tọa độ ROI không hợp lệ: roi_x={roi_x}, roi_y={roi_y}. "
            "Tọa độ phải >= 0."
        )
    
    if roi_x + roi_width > w or roi_y + roi_height > h:
        raise ValueError(
            f"ROI vượt quá giới hạn ảnh. "
            f"Ảnh: {w}x{h}, ROI: ({roi_x}, {roi_y}) -> ({roi_x + roi_width}, {roi_y + roi_height})"
        )
    
    # Bước 2 - Cắt vùng mã đề
    region = warped_image[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    
    # Bước 3 - Return vùng đã cắt
    return region


def read_exam_code(exam_code_region: np.ndarray,
                  num_digits: int = 3,
                  choices_per_digit: int = 10,
                  threshold_method: str = "adaptive") -> str:
    """
    Đọc mã đề từ vùng ROI đã cắt.
    
    Args:
        exam_code_region (np.ndarray): Ảnh vùng chứa mã đề (từ extract_exam_code_region)
        num_digits (int): Số chữ số của mã đề. Mặc định 3 (101, 102, ...).
        choices_per_digit (int): Số lựa chọn mỗi chữ số (0-9 = 10). Mặc định 10.
        threshold_method (str): Phương pháp phân ngưỡng ("adaptive", "otsu", "binary")
    
    Returns:
        str: Mã đề dạng string, ví dụ: "101", "102", "103"
    
    Raises:
        ValueError: Nếu không đọc được mã đề (không có ô nào được tô hoặc tô nhiều ô)
    
    Notes:
        - Thuật toán tương tự grader.calculate_score()
        - Chia vùng thành lưới num_digits x choices_per_digit
        - Đếm pixel trắng để xác định ô được tô
        - Ghép các chữ số thành mã đề
        - Nếu 1 chữ số có nhiều hơn 1 ô được tô -> raise ValueError
    
    Algorithm:
        1. Chuyển sang ảnh xám (nếu cần)
        2. Phân ngưỡng (adaptive/otsu/binary)
        3. Tính kích thước mỗi bubble:
           - bubble_height = exam_code_region.shape[0] // choices_per_digit
           - bubble_width = exam_code_region.shape[1] // num_digits
        4. Lặp qua từng chữ số (cột):
           - Lặp qua từng lựa chọn 0-9 (hàng)
           - Đếm pixel trắng trong mỗi bubble
           - Tìm bubble có pixel nhiều nhất
           - Chuyển index thành chữ số (0-9)
        5. Ghép các chữ số thành string
        6. Return mã đề
    
    Examples:
        >>> exam_code = read_exam_code(exam_code_region, num_digits=3)
        >>> print(exam_code)
        "102"
        
        >>> # Mã đề 4 chữ số
        >>> exam_code = read_exam_code(exam_code_region, num_digits=4)
        >>> print(exam_code)
        "1024"
    """
    # Bước 1 - Chuyển sang ảnh xám nếu cần
    if exam_code_region.ndim == 3:
        gray = cv2.cvtColor(exam_code_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = exam_code_region.copy()
    
    # Bước 1.5 - Làm sạch nhiễu bằng morphological operations
    # Áp dụng Gaussian blur nhẹ để giảm nhiễu
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Bước 2 - Phân ngưỡng
    if threshold_method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )
    elif threshold_method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif threshold_method == "binary":
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    else:
        raise ValueError(
            f"Phương pháp phân ngưỡng không hợp lệ: '{threshold_method}'. "
            "Các giá trị hợp lệ: 'adaptive', 'otsu', 'binary'"
        )
    
    # Bước 2.5 - Làm sạch nhiễu sau phân ngưỡng
    # Loại bỏ các điểm nhiễu nhỏ bằng morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Bước 3 - Tính kích thước mỗi bubble
    bubble_height = binary.shape[0] // choices_per_digit  # 10 hàng (0-9)
    bubble_width = binary.shape[1] // num_digits          # 3 cột (3 chữ số)
    
    if bubble_height == 0 or bubble_width == 0:
        raise ValueError(
            f"Kích thước bubble không hợp lệ. "
            f"ROI: {binary.shape}, num_digits: {num_digits}, choices_per_digit: {choices_per_digit}"
        )
    
    # Bước 4 - Đọc từng chữ số
    exam_code = ""
    
    for digit_idx in range(num_digits):  # Lặp qua từng cột (chữ số)
        pixel_counts = []
        
        for choice_idx in range(choices_per_digit):  # Lặp qua từng hàng (0-9)
            # Cắt bubble
            y1 = choice_idx * bubble_height
            y2 = (choice_idx + 1) * bubble_height
            x1 = digit_idx * bubble_width
            x2 = (digit_idx + 1) * bubble_width
            bubble = binary[y1:y2, x1:x2]
            
            # Đếm pixel trắng
            count = cv2.countNonZero(bubble)
            pixel_counts.append(count)
        
        # Tìm ô được tô (pixel nhiều nhất)
        max_idx = np.argmax(pixel_counts)
        max_count = pixel_counts[max_idx]
        
        # Kiểm tra có đúng 1 ô được tô không
        bubble_area = bubble_height * bubble_width
        threshold = bubble_area * 0.10  # 10% diện tích (giảm từ 15% để linh hoạt hơn)
        
        if max_count < threshold:
            raise ValueError(
                f"Chữ số thứ {digit_idx + 1} không có ô nào được tô. "
                f"Pixel count tối đa: {max_count}, ngưỡng: {threshold:.0f}"
            )
        
        # Kiểm tra có tô nhiều ô không (ô thứ 2 phải < 40% của ô thứ nhất)
        # Giảm từ 50% xuống 40% để chặt chẽ hơn
        sorted_counts = sorted(pixel_counts, reverse=True)
        if len(sorted_counts) > 1 and sorted_counts[1] > max_count * 0.40:
            raise ValueError(
                f"Chữ số thứ {digit_idx + 1} có nhiều ô được tô. "
                f"Pixel counts: {sorted_counts[:2]}"
            )
        
        # Ghép chữ số vào mã đề
        exam_code += str(max_idx)
    
    # Bước 5 - Return mã đề
    return exam_code


# ============================================================================
# ĐỌC MÃ SINH VIÊN (STUDENT ID) - TÙY CHỌN
# ============================================================================

def extract_student_id_region(warped_image: np.ndarray,
                              roi_x: int = 161,
                              roi_y: int = 439,
                              roi_width: int = 207,
                              roi_height: int = 323) -> np.ndarray:
    """
    Cắt vùng chứa mã sinh viên (thường ở góc trên bên trái).
    
    Args:
        warped_image: Ảnh đã nắn chỉnh
        roi_x: Tọa độ x góc trên-trái
        roi_y: Tọa độ y góc trên-trái
        roi_width: Chiều rộng vùng
        roi_height: Chiều cao vùng
    
    Returns:
        Ảnh vùng ROI chứa mã sinh viên
    
    Raises:
        ValueError: Nếu ROI vượt quá giới hạn ảnh
    
    Notes:
        - Mã sinh viên thường có 8-10 chữ số
        - Tương tự mã đề, mỗi chữ số có 10 ô (0-9)
    """
    # Sử dụng lại logic của extract_exam_code_region
    h, w = warped_image.shape[:2]
    
    if roi_x < 0 or roi_y < 0:
        raise ValueError(
            f"Tọa độ ROI không hợp lệ: roi_x={roi_x}, roi_y={roi_y}. "
            "Tọa độ phải >= 0."
        )
    
    if roi_x + roi_width > w or roi_y + roi_height > h:
        raise ValueError(
            f"ROI vượt quá giới hạn ảnh. "
            f"Ảnh: {w}x{h}, ROI: ({roi_x}, {roi_y}) -> ({roi_x + roi_width}, {roi_y + roi_height})"
        )
    
    region = warped_image[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    return region


def read_student_id(student_id_region: np.ndarray,
                   num_digits: int = 8,
                   threshold_method: str = "adaptive") -> str:
    """
    Đọc mã sinh viên từ vùng ROI.
    
    Args:
        student_id_region: Ảnh vùng chứa mã sinh viên
        num_digits: Số chữ số của mã sinh viên (mặc định 8)
        threshold_method: Phương pháp phân ngưỡng
    
    Returns:
        Mã sinh viên dạng string
    
    Raises:
        ValueError: Nếu không đọc được mã sinh viên
    
    Notes:
        - Thuật toán giống read_exam_code()
    """
    # Sử dụng lại logic của read_exam_code
    return read_exam_code(
        student_id_region,
        num_digits=num_digits,
        choices_per_digit=10,
        threshold_method=threshold_method
    )


# ============================================================================
# HÀM TIỆN ÍCH
# ============================================================================

def visualize_exam_code_region(warped_image: np.ndarray,
                               roi_x: int = 406,
                               roi_y: int = 437,
                               roi_width: int = 112,
                               roi_height: int = 322) -> np.ndarray:
    """
    Vẽ khung ROI lên ảnh để kiểm tra vị trí mã đề.
    
    Args:
        warped_image: Ảnh đã nắn chỉnh
        roi_x, roi_y, roi_width, roi_height: Tọa độ ROI
    
    Returns:
        Ảnh có vẽ khung ROI màu xanh lá
    
    Notes:
        - Dùng để debug và xác định tọa độ ROI chính xác
        - Lưu ảnh ra file để kiểm tra
    """
    # Bước 1 - Copy ảnh để không thay đổi ảnh gốc
    vis_image = warped_image.copy()
    
    # Chuyển sang màu nếu là ảnh xám
    if vis_image.ndim == 2:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
    
    # Bước 2 - Vẽ khung ROI
    cv2.rectangle(
        vis_image,
        (roi_x, roi_y),
        (roi_x + roi_width, roi_y + roi_height),
        (0, 255, 0),  # Màu xanh lá
        2  # Độ dày
    )
    
    # Bước 3 - Thêm text "Exam Code"
    cv2.putText(
        vis_image,
        "Exam Code",
        (roi_x, roi_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )
    
    # Bước 4 - Return ảnh
    return vis_image


def visualize_student_id_region(warped_image: np.ndarray,
                                roi_x: int = 161,
                                roi_y: int = 439,
                                roi_width: int = 207,
                                roi_height: int = 323) -> np.ndarray:
    """
    Vẽ khung ROI lên ảnh để kiểm tra vị trí mã sinh viên.
    
    Args:
        warped_image: Ảnh đã nắn chỉnh
        roi_x, roi_y, roi_width, roi_height: Tọa độ ROI
    
    Returns:
        Ảnh có vẽ khung ROI màu xanh dương
    """
    vis_image = warped_image.copy()
    
    if vis_image.ndim == 2:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
    
    cv2.rectangle(
        vis_image,
        (roi_x, roi_y),
        (roi_x + roi_width, roi_y + roi_height),
        (255, 0, 0),  # Màu xanh dương
        2
    )
    
    cv2.putText(
        vis_image,
        "Student ID",
        (roi_x, roi_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )
    
    return vis_image


def visualize_all_regions(warped_image: np.ndarray,
                         exam_code_roi: Tuple[int, int, int, int] = (406, 437, 112, 322),
                         student_id_roi: Tuple[int, int, int, int] = (161, 439, 207, 323),
                         answer_roi: Tuple[int, int, int, int] = (157, 797, 391, 319)) -> np.ndarray:
    """
    Vẽ tất cả các vùng ROI lên ảnh để kiểm tra.
    
    Args:
        warped_image: Ảnh đã nắn chỉnh
        exam_code_roi: (x, y, width, height) của vùng mã đề
        student_id_roi: (x, y, width, height) của vùng MSSV
        answer_roi: (x, y, width, height) của vùng đáp án
    
    Returns:
        Ảnh có vẽ tất cả các khung ROI
    """
    vis_image = warped_image.copy()
    
    if vis_image.ndim == 2:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
    
    # Vẽ vùng mã đề (xanh lá)
    x, y, w, h = exam_code_roi
    cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(vis_image, "Exam Code", (x, y - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Vẽ vùng MSSV (xanh dương)
    x, y, w, h = student_id_roi
    cv2.rectangle(vis_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(vis_image, "Student ID", (x, y - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Vẽ vùng đáp án (đỏ)
    x, y, w, h = answer_roi
    cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(vis_image, "Answers", (x, y - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return vis_image
