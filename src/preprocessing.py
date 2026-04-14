"""
Module tiền xử lý ảnh cho hệ thống chấm trắc nghiệm tự động (OMR).

Module này chứa các hàm cơ bản để đọc ảnh, chuyển đổi không gian màu,
và áp dụng các bộ lọc khử nhiễu.
"""

import os

import cv2
import numpy as np
from typing import Optional


def doc_anh(duong_dan: str) -> np.ndarray:
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
    # Bước 1 - Kiểm tra file tồn tại trước khi gọi OpenCV
    # cv2.imread() trả None thầm lặng cho cả 2 trường hợp:
    # không tồn tại VÀ không phải ảnh → cần phân biệt để raise đúng exception
    if not os.path.exists(duong_dan):
        raise FileNotFoundError(
            f"Không tìm thấy file ảnh: '{duong_dan}'"
        )

    # Bước 2 - Đọc ảnh bằng cv2.imread()
    # Mặc định cv2.IMREAD_COLOR: đọc ảnh màu BGR 3 channel
    anh = cv2.imread(duong_dan)

    # Bước 3 - Kiểm tra ảnh có được đọc thành công không
    # File tồn tại nhưng cv2 không decode được → file bị hỏng hoặc không phải ảnh
    if anh is None:
        raise ValueError(
            f"File không phải ảnh hợp lệ hoặc bị hỏng: '{duong_dan}'"
        )

    # Bước 4 - Trả về ảnh dạng np.ndarray shape (H, W, 3), dtype uint8
    return anh


def chuyen_xam(anh: np.ndarray) -> np.ndarray:
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
    # Bước 1 - Guard clause: nếu ảnh đã là xám (2D array) → return trực tiếp
    # Tránh crash khi hàm được gọi lại trong quá trình debug pipeline
    if anh.ndim == 2:
        return anh

    # Bước 2 - Chuyển đổi BGR → Grayscale theo chuẩn ITU-R BT.601
    # Công thức: Gray = 0.299*R + 0.587*G + 0.114*B
    # OpenCV tối ưu nội bộ bằng SIMD, nhanh hơn tự tính bằng NumPy
    anh_xam = cv2.cvtColor(anh, cv2.COLOR_BGR2GRAY)

    # Bước 3 - Trả về ảnh xám shape (H, W), dtype uint8
    return anh_xam


def loc_nhieu(anh_xam: np.ndarray,
              loai_loc: str = "gaussian",
              kich_thuoc: int = 5) -> np.ndarray:
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
    # Bước 1 - Validate kich_thuoc: phải là số lẻ dương
    # Kernel cần có 1 pixel trung tâm chính xác → số chẵn không có tâm
    if kich_thuoc <= 0 or kich_thuoc % 2 == 0:
        raise ValueError(
            f"kich_thuoc phải là số lẻ dương, nhận được: {kich_thuoc}"
        )

    # Bước 2 - Validate loai_loc
    cac_bo_loc_hop_le = ("gaussian", "median", "bilateral")
    if loai_loc not in cac_bo_loc_hop_le:
        raise ValueError(
            f"loai_loc không hợp lệ: '{loai_loc}'. "
            f"Các giá trị hợp lệ: {cac_bo_loc_hop_le}"
        )

    # Bước 3 - Gaussian Blur: tích chập với kernel Gaussian
    # Phù hợp nhất cho nhiễu Gaussian — loại nhiễu phổ biến nhất trong ảnh scan
    # sigma=0 → OpenCV tự tính từ kich_thuoc: sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    if loai_loc == "gaussian":
        return cv2.GaussianBlur(anh_xam, (kich_thuoc, kich_thuoc), 0)

    # Bước 4 - Median Blur: thay pixel bằng giá trị trung vị trong vùng lân cận
    # Loại bỏ được nhiễu muối tiêu (salt-and-pepper) vì median loại cực trị
    if loai_loc == "median":
        return cv2.medianBlur(anh_xam, kich_thuoc)

    # Bước 5 - Bilateral Filter: kết hợp Gaussian không gian + Gaussian cường độ
    # Edge-preserving: pixel có cường độ khác biệt lớn (= cạnh) không bị trộn
    # sigmaColor=75: chênh lệch cường độ > 75 → coi là cạnh, không blur qua
    # sigmaSpace=75: phạm vi ảnh hưởng không gian của filter
    return cv2.bilateralFilter(anh_xam, kich_thuoc, sigmaColor=75, sigmaSpace=75)
