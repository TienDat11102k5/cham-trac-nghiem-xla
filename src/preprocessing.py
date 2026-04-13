"""
Module tiền xử lý ảnh cho hệ thống chấm trắc nghiệm tự động (OMR).

Module này chứa các hàm cơ bản để đọc ảnh, chuyển đổi không gian màu,
và áp dụng các bộ lọc khử nhiễu.
"""

import cv2
import numpy as np
from typing import Optional


def load_image(image_path: str) -> np.ndarray:
    """
    Đọc ảnh từ đường dẫn file.
    
    Args:
        image_path (str): Đường dẫn tuyệt đối hoặc tương đối đến file ảnh.
                         Hỗ trợ các định dạng: .jpg, .jpeg, .png, .bmp
    
    Returns:
        np.ndarray: Ảnh dạng numpy array với shape (height, width, channels).
                    Channels theo thứ tự BGR (Blue, Green, Red) - chuẩn OpenCV.
    
    Raises:
        FileNotFoundError: Nếu file ảnh không tồn tại.
        ValueError: Nếu file không phải là ảnh hợp lệ.
    
    Examples:
        >>> image = load_image("data/test_sheet_01.jpg")
        >>> print(image.shape)
        (1200, 800, 3)
    """
    # TODO: Bước 1 - Sử dụng cv2.imread() để đọc ảnh từ image_path
    # TODO: Bước 2 - Kiểm tra xem ảnh có được đọc thành công không (image is not None)
    # TODO: Bước 3 - Nếu ảnh None, raise FileNotFoundError với thông báo rõ ràng
    # TODO: Bước 4 - Return ảnh đã đọc được
    raise NotImplementedError


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Chuyển đổi ảnh màu (BGR) sang ảnh xám (Grayscale).
    
    Args:
        image (np.ndarray): Ảnh màu đầu vào với shape (height, width, 3).
                           Channels theo thứ tự BGR.
    
    Returns:
        np.ndarray: Ảnh xám với shape (height, width).
                    Giá trị pixel trong khoảng [0, 255].
    
    Notes:
        - Ảnh xám giúp giảm độ phức tạp tính toán (từ 3 channels xuống 1 channel).
        - Công thức chuyển đổi: Gray = 0.299*R + 0.587*G + 0.114*B
    
    Examples:
        >>> color_image = load_image("test.jpg")
        >>> gray_image = convert_to_grayscale(color_image)
        >>> print(gray_image.shape)
        (1200, 800)
    """
    # TODO: Bước 1 - Sử dụng cv2.cvtColor() với flag cv2.COLOR_BGR2GRAY
    # TODO: Bước 2 - Return ảnh xám đã chuyển đổi
    raise NotImplementedError


def apply_noise_filter(gray_image: np.ndarray, 
                       filter_type: str = "gaussian",
                       kernel_size: int = 5) -> np.ndarray:
    """
    Áp dụng bộ lọc khử nhiễu cho ảnh xám.
    
    Args:
        gray_image (np.ndarray): Ảnh xám đầu vào với shape (height, width).
        filter_type (str, optional): Loại bộ lọc. Mặc định là "gaussian".
                                    Các giá trị hợp lệ: "gaussian", "median", "bilateral"
        kernel_size (int, optional): Kích thước kernel (phải là số lẻ). Mặc định là 5.
    
    Returns:
        np.ndarray: Ảnh đã được làm mờ/khử nhiễu với cùng shape như ảnh đầu vào.
    
    Notes:
        - Gaussian Blur: Tốt cho nhiễu Gaussian, làm mờ đều.
        - Median Filter: Tốt cho nhiễu muối tiêu (salt-and-pepper noise).
        - Bilateral Filter: Giữ được cạnh sắc nét trong khi làm mờ vùng phẳng.
    
    Raises:
        ValueError: Nếu filter_type không hợp lệ hoặc kernel_size là số chẵn.
    
    Examples:
        >>> gray = convert_to_grayscale(image)
        >>> blurred = apply_noise_filter(gray, filter_type="gaussian", kernel_size=5)
    """
    # TODO: Bước 1 - Kiểm tra kernel_size phải là số lẻ và > 0
    # TODO: Bước 2 - Kiểm tra filter_type có hợp lệ không
    # TODO: Bước 3 - Nếu filter_type == "gaussian":
    #                Sử dụng cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
    # TODO: Bước 4 - Nếu filter_type == "median":
    #                Sử dụng cv2.medianBlur(gray_image, kernel_size)
    # TODO: Bước 5 - Nếu filter_type == "bilateral":
    #                Sử dụng cv2.bilateralFilter(gray_image, kernel_size, sigmaColor, sigmaSpace)
    #                với sigmaColor=75, sigmaSpace=75
    # TODO: Bước 6 - Return ảnh đã được lọc
    raise NotImplementedError
