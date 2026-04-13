"""
Module chấm điểm cho hệ thống chấm trắc nghiệm tự động (OMR).

Module này chứa các hàm để trích xuất vùng đáp án, phân đoạn các ô trắc nghiệm,
và tính điểm dựa trên đáp án chuẩn.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional


def extract_bubble_grid(warped_image: np.ndarray,
                        roi_x: int = 100,
                        roi_y: int = 200,
                        roi_width: int = 600,
                        roi_height: int = 800) -> np.ndarray:
    """
    Cắt lấy vùng chứa các ô đáp án (Region of Interest - ROI) từ ảnh đã nắn chỉnh.
    
    Args:
        warped_image (np.ndarray): Ảnh đã được nắn chỉnh, shape (height, width) hoặc (height, width, 3).
        roi_x (int, optional): Tọa độ x góc trên-trái của ROI. Mặc định là 100.
        roi_y (int, optional): Tọa độ y góc trên-trái của ROI. Mặc định là 200.
        roi_width (int, optional): Chiều rộng của ROI. Mặc định là 600.
        roi_height (int, optional): Chiều cao của ROI. Mặc định là 800.
    
    Returns:
        np.ndarray: Ảnh vùng ROI chứa các ô đáp án, shape (roi_height, roi_width) hoặc (roi_height, roi_width, 3).
    
    Notes:
        - ROI là vùng chứa tất cả các ô trắc nghiệm (bubbles) trên tờ giấy thi.
        - Tọa độ ROI có thể cần điều chỉnh tùy theo template của đề thi.
        - Nên xác định ROI bằng cách phân tích một vài mẫu ảnh trước.
    
    Examples:
        >>> grid = extract_bubble_grid(warped_image, roi_x=100, roi_y=200, roi_width=600, roi_height=800)
        >>> print(grid.shape)
        (800, 600)
    """
    # TODO: Bước 1 - Kiểm tra tọa độ ROI có nằm trong giới hạn ảnh không
    #                (roi_x + roi_width <= warped_image.shape[1] và roi_y + roi_height <= warped_image.shape[0])
    # TODO: Bước 2 - Cắt ảnh theo ROI: grid = warped_image[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
    # TODO: Bước 3 - Return vùng grid đã cắt
    raise NotImplementedError


def segment_bubbles(grid_image: np.ndarray,
                    threshold_method: str = "adaptive") -> np.ndarray:
    """
    Áp dụng phân ngưỡng (Thresholding) để chuyển ảnh sang nhị phân (trắng/đen).
    
    Args:
        grid_image (np.ndarray): Ảnh vùng ROI chứa các ô đáp án, shape (height, width).
                                Nên là ảnh xám.
        threshold_method (str, optional): Phương pháp phân ngưỡng. Mặc định là "adaptive".
                                         Các giá trị hợp lệ: "adaptive", "otsu", "binary"
    
    Returns:
        np.ndarray: Ảnh nhị phân với shape (height, width).
                    Pixel trắng (255) là nền, pixel đen (0) là vùng được tô (đáp án đã chọn).
    
    Notes:
        - Adaptive Threshold: Tốt khi ảnh có độ sáng không đều.
        - Otsu's Method: Tự động tìm ngưỡng tối ưu dựa trên histogram.
        - Binary Threshold: Sử dụng ngưỡng cố định (cần điều chỉnh thủ công).
    
    Raises:
        ValueError: Nếu threshold_method không hợp lệ.
    
    Examples:
        >>> binary = segment_bubbles(grid_image, threshold_method="adaptive")
        >>> cv2.imshow("Binary", binary)
    """
    # TODO: Bước 1 - Nếu grid_image là ảnh màu, chuyển sang xám bằng cv2.cvtColor()
    # TODO: Bước 2 - Kiểm tra threshold_method có hợp lệ không
    # TODO: Bước 3 - Nếu threshold_method == "adaptive":
    #                Sử dụng cv2.adaptiveThreshold() với:
    #                - Method: cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    #                - Type: cv2.THRESH_BINARY_INV (đảo ngược: vùng tô thành trắng)
    #                - Block size: 11
    #                - C constant: 2
    # TODO: Bước 4 - Nếu threshold_method == "otsu":
    #                Sử dụng cv2.threshold() với flag cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    # TODO: Bước 5 - Nếu threshold_method == "binary":
    #                Sử dụng cv2.threshold() với ngưỡng cố định (ví dụ: 127)
    # TODO: Bước 6 - (Tùy chọn) Áp dụng morphological operations để làm sạch nhiễu:
    #                - cv2.erode() để loại bỏ nhiễu nhỏ
    #                - cv2.dilate() để khôi phục kích thước vùng tô
    # TODO: Bước 7 - Return ảnh nhị phân
    raise NotImplementedError


def calculate_score(segmented_image: np.ndarray,
                   answer_key: Dict[int, str],
                   num_questions: int = 40,
                   choices_per_question: int = 4) -> Tuple[int, float, Dict[int, str]]:
    """
    Đếm điểm ảnh, so sánh với đáp án chuẩn và trả về điểm số.
    
    Args:
        segmented_image (np.ndarray): Ảnh nhị phân đã phân đoạn, shape (height, width).
        answer_key (Dict[int, str]): Đáp án chuẩn. Key là số câu hỏi (1-indexed), Value là đáp án ('A', 'B', 'C', 'D').
                                     Ví dụ: {1: 'A', 2: 'C', 3: 'B', ...}
        num_questions (int, optional): Tổng số câu hỏi. Mặc định là 40.
        choices_per_question (int, optional): Số lựa chọn mỗi câu (A, B, C, D = 4). Mặc định là 4.
    
    Returns:
        Tuple[int, float, Dict[int, str]]: Bộ 3 giá trị:
            - correct_count (int): Số câu trả lời đúng.
            - score (float): Điểm số (thang điểm 10). Công thức: (correct_count / num_questions) * 10
            - student_answers (Dict[int, str]): Đáp án của học sinh. Key là số câu, Value là đáp án đã chọn.
    
    Notes:
        - Mỗi ô trắc nghiệm (bubble) được xác định bằng cách chia đều segmented_image thành lưới.
        - Đếm số pixel trắng trong mỗi bubble để xác định ô nào được tô.
        - Bubble có số pixel trắng nhiều nhất trong mỗi câu hỏi là đáp án được chọn.
        - Nếu không có bubble nào được tô hoặc tô nhiều hơn 1 bubble, câu đó tính là sai.
    
    Examples:
        >>> answer_key = {1: 'A', 2: 'C', 3: 'B', 4: 'D'}
        >>> correct, score, answers = calculate_score(binary_image, answer_key, num_questions=4)
        >>> print(f"Điểm: {score}/10, Số câu đúng: {correct}/{len(answer_key)}")
        Điểm: 7.5/10, Số câu đúng: 3/4
    """
    # TODO: Bước 1 - Tính kích thước mỗi bubble:
    #                bubble_height = segmented_image.shape[0] // num_questions
    #                bubble_width = segmented_image.shape[1] // choices_per_question
    # TODO: Bước 2 - Khởi tạo dictionary để lưu đáp án học sinh: student_answers = {}
    # TODO: Bước 3 - Khởi tạo biến đếm số câu đúng: correct_count = 0
    # TODO: Bước 4 - Lặp qua từng câu hỏi (i từ 0 đến num_questions-1):
    #                4.1 - Khởi tạo list để lưu số pixel trắng của mỗi lựa chọn: pixel_counts = []
    #                4.2 - Lặp qua từng lựa chọn (j từ 0 đến choices_per_question-1):
    #                      - Cắt vùng bubble: bubble = segmented_image[i*bubble_height:(i+1)*bubble_height,
    #                                                                   j*bubble_width:(j+1)*bubble_width]
    #                      - Đếm số pixel trắng: count = cv2.countNonZero(bubble)
    #                      - Thêm count vào pixel_counts
    #                4.3 - Tìm index của bubble có pixel_counts lớn nhất: max_index = np.argmax(pixel_counts)
    #                4.4 - Kiểm tra xem có đúng 1 bubble được tô không (pixel_counts[max_index] > threshold)
    #                      threshold có thể là 50% diện tích bubble
    #                4.5 - Chuyển max_index thành chữ cái: choices = ['A', 'B', 'C', 'D']
    #                      student_answer = choices[max_index]
    #                4.6 - Lưu vào student_answers: student_answers[i+1] = student_answer
    #                4.7 - So sánh với answer_key: nếu student_answer == answer_key.get(i+1), tăng correct_count
    # TODO: Bước 5 - Tính điểm: score = (correct_count / num_questions) * 10
    # TODO: Bước 6 - Return (correct_count, score, student_answers)
    raise NotImplementedError
