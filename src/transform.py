"""
Module biến đổi hình học cho hệ thống chấm trắc nghiệm tự động (OMR).

Module này chứa các hàm để phát hiện biên, tìm contour của tờ giấy thi,
và thực hiện phép biến đổi phối cảnh (perspective transform) để nắn chỉnh ảnh.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


def tim_canh(anh_mo: np.ndarray,
             nguong_thap: int = 50,
             nguong_cao: int = 150) -> np.ndarray:
    """
    Phát hiện biên trong ảnh sử dụng thuật toán Canny Edge Detection.
    
    Args:
        blurred_image (np.ndarray): Ảnh xám đã được làm mờ, shape (height, width).
        low_threshold (int, optional): Ngưỡng dưới cho Canny. Mặc định là 50.
        high_threshold (int, optional): Ngưỡng trên cho Canny. Mặc định là 150.
    
    Returns:
        np.ndarray: Ảnh nhị phân chứa các cạnh được phát hiện.
                    Pixel trắng (255) là cạnh, pixel đen (0) là nền.
                    Shape: (height, width)
    
    Notes:
        - Tỷ lệ low:high nên là 1:2 hoặc 1:3 để có kết quả tốt nhất.
        - Ảnh đầu vào nên được làm mờ trước để giảm nhiễu.
        - Canny sử dụng đạo hàm Sobel để tính gradient.
    
    Examples:
        >>> edges = detect_edges(blurred_image, low_threshold=50, high_threshold=150)
        >>> cv2.imshow("Edges", edges)
    """
    # TODO: Bước 1 - Sử dụng cv2.Canny() với các tham số:
    #                anh_mo, nguong_thap, nguong_cao
    # TODO: Bước 2 - Return ảnh biên (edge map)
    raise NotImplementedError


def tim_goc_giay(anh_canh: np.ndarray) -> np.ndarray:
    """
    Tìm 4 góc của tờ giấy thi từ ảnh biên.
    
    Args:
        edges (np.ndarray): Ảnh biên nhị phân từ Canny, shape (height, width).
    
    Returns:
        np.ndarray: Mảng chứa tọa độ 4 góc của tờ giấy, shape (4, 2).
                    Thứ tự: [top-left, top-right, bottom-right, bottom-left]
                    Mỗi điểm có dạng [x, y] với kiểu dữ liệu float32.
    
    Raises:
        ValueError: Nếu không tìm thấy contour hợp lệ (tờ giấy không có trong ảnh).
    
    Notes:
        - Hàm tìm contour lớn nhất (có diện tích lớn nhất).
        - Sử dụng xấp xỉ đa giác (polygon approximation) để giảm số điểm.
        - Giả định tờ giấy thi là đối tượng lớn nhất trong ảnh.
    
    Examples:
        >>> corners = find_document_corners(edges)
        >>> print(corners)
        [[120.5, 80.3], [680.2, 75.1], [690.8, 920.4], [110.3, 925.7]]
    """
    # TODO: Bước 1 - Sử dụng cv2.findContours() để tìm tất cả contours
    #                Mode: cv2.RETR_EXTERNAL (chỉ lấy contour ngoài cùng)
    #                Method: cv2.CHAIN_APPROX_SIMPLE (nén contour)
    # TODO: Bước 2 - Sắp xếp contours theo diện tích giảm dần: sorted(contours, key=cv2.contourArea, reverse=True)
    # TODO: Bước 3 - Lấy contour lớn nhất (contours[0])
    # TODO: Bước 4 - Tính chu vi contour: perimeter = cv2.arcLength(contour, closed=True)
    # TODO: Bước 5 - Xấp xỉ contour thành đa giác: approx = cv2.approxPolyDP(contour, epsilon=0.02*perimeter, closed=True)
    # TODO: Bước 6 - Kiểm tra xem approx có đúng 4 điểm không (len(approx) == 4)
    # TODO: Bước 7 - Nếu không có 4 điểm, raise ValueError("Không tìm thấy tờ giấy thi")
    # TODO: Bước 8 - Reshape approx thành shape (4, 2) và convert sang float32
    # TODO: Bước 9 - Sắp xếp 4 điểm theo thứ tự: top-left, top-right, bottom-right, bottom-left
    #                Gợi ý: Tính tổng (x+y) và hiệu (y-x) để xác định vị trí
    # TODO: Bước 10 - Return mảng 4 góc đã sắp xếp
    raise NotImplementedError


def nan_chinh_anh(anh: np.ndarray, 
                  cac_goc: np.ndarray,
                  chieu_rong: int = 800,
                  chieu_cao: int = 1200) -> np.ndarray:
    """
    Áp dụng phép biến đổi phối cảnh để nắn chỉnh ảnh từ góc nghiêng về mặt phẳng chuẩn.
    
    Args:
        image (np.ndarray): Ảnh gốc (màu hoặc xám) cần nắn chỉnh.
        corners (np.ndarray): Tọa độ 4 góc của tờ giấy trong ảnh gốc, shape (4, 2).
                             Thứ tự: [top-left, top-right, bottom-right, bottom-left]
        output_width (int, optional): Chiều rộng ảnh đầu ra. Mặc định là 800 pixels.
        output_height (int, optional): Chiều cao ảnh đầu ra. Mặc định là 1200 pixels.
    
    Returns:
        np.ndarray: Ảnh đã được nắn chỉnh với shape (output_height, output_width, channels).
                    Tờ giấy thi giờ đã ở dạng nhìn thẳng (bird's eye view).
    
    Notes:
        - Phép biến đổi phối cảnh (Perspective Transform) ánh xạ 4 điểm nguồn
          thành 4 điểm đích tạo thành hình chữ nhật.
        - Tỷ lệ output_width:output_height nên phù hợp với tờ giấy thật (ví dụ A4: 210x297mm).
    
    Examples:
        >>> warped = apply_perspective_transform(image, corners, output_width=800, output_height=1200)
        >>> cv2.imshow("Warped", warped)
    """
    # TODO: Bước 1 - Định nghĩa 4 điểm đích (destination points) tạo thành hình chữ nhật:
    #                dst_points = np.array([
    #                    [0, 0],                              # top-left
    #                    [chieu_rong - 1, 0],                 # top-right
    #                    [chieu_rong - 1, chieu_cao - 1],     # bottom-right
    #                    [0, chieu_cao - 1]                   # bottom-left
    #                ], dtype=np.float32)
    # TODO: Bước 2 - Tính ma trận biến đổi phối cảnh:
    #                matrix = cv2.getPerspectiveTransform(cac_goc, dst_points)
    # TODO: Bước 3 - Áp dụng phép biến đổi:
    #                warped = cv2.warpPerspective(anh, matrix, (chieu_rong, chieu_cao))
    # TODO: Bước 4 - Return ảnh đã nắn chỉnh
    raise NotImplementedError
