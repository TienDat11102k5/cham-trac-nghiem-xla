"""
Module biến đổi hình học cho hệ thống chấm trắc nghiệm tự động (OMR).

Module này chứa các hàm để phát hiện biên, tìm contour của tờ giấy thi,
và thực hiện phép biến đổi phối cảnh (perspective transform) để nắn chỉnh ảnh.

Thành viên phụ trách: Thành viên 3
Nhiệm vụ: Biến đổi hình học - Tìm góc và nắn chỉnh ảnh

Functions:
    tim_canh: Phát hiện biên bằng Canny Edge Detection
    tim_goc_giay: Tìm 4 góc của tờ giấy thi
    nan_chinh_anh: Nắn chỉnh ảnh bằng Perspective Transform
    
Examples:
    >>> from src.preprocessing import doc_anh, chuyen_xam, loc_nhieu
    >>> from src.transform import tim_canh, tim_goc_giay, nan_chinh_anh
    >>> 
    >>> # Pipeline xử lý
    >>> anh = doc_anh("data/raw/test_sheet_01.jpg")
    >>> anh_xam = chuyen_xam(anh)
    >>> anh_mo = loc_nhieu(anh_xam, "gaussian", 5)
    >>> canh = tim_canh(anh_mo)
    >>> cac_goc = tim_goc_giay(canh)
    >>> anh_thang = nan_chinh_anh(anh, cac_goc)
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


def tim_canh(anh_mo: np.ndarray,
             nguong_thap: int = 30,
             nguong_cao: int = 100) -> np.ndarray:
    """
    Phát hiện biên trong ảnh sử dụng thuật toán Canny Edge Detection.
    
    Args:
        anh_mo (np.ndarray): Ảnh xám đã được làm mờ, shape (height, width).
        nguong_thap (int, optional): Ngưỡng dưới cho Canny. Mặc định là 30.
        nguong_cao (int, optional): Ngưỡng trên cho Canny. Mặc định là 100.
    
    Returns:
        np.ndarray: Ảnh nhị phân chứa các cạnh được phát hiện.
    """
    canh = cv2.Canny(anh_mo, nguong_thap, nguong_cao)
    return canh


def tim_goc_giay(anh_canh: np.ndarray) -> np.ndarray:
    """
    Tìm 4 góc của tờ giấy thi từ ảnh biên.
    
    Sử dụng morphology để làm nổi bật viền tờ giấy, sau đó tìm contour lớn nhất.
    
    Args:
        anh_canh (np.ndarray): Ảnh biên nhị phân từ Canny, shape (height, width).
    
    Returns:
        np.ndarray: Mảng chứa tọa độ 4 góc của tờ giấy, shape (4, 2).
    
    Raises:
        ValueError: Nếu không tìm thấy tờ giấy.
    """
    # Bước 1 - Áp dụng morphology nhẹ để đóng các khoảng trống nhỏ
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    anh_dong = cv2.morphologyEx(anh_canh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Bước 2 - Dilate nhẹ để làm dày viền
    anh_dilate = cv2.dilate(anh_dong, kernel, iterations=1)
    
    # Bước 3 - Tìm contours
    contours, _ = cv2.findContours(anh_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError(
            "Không tìm thấy tờ giấy thi trong ảnh. "
            "Vui lòng kiểm tra:\n"
            "  - Tờ giấy có nằm hoàn toàn trong khung ảnh không?\n"
            "  - Độ tương phản giữa giấy và nền có đủ rõ không?\n"
            "  - Ảnh có bị mờ hoặc nhiễu quá nhiều không?"
        )
    
    # Bước 4 - Sắp xếp theo diện tích
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    h_anh, w_anh = anh_canh.shape
    image_area = h_anh * w_anh
    
    # Bước 5 - Tìm contour đầu tiên có 4 góc và diện tích đủ lớn
    # Ưu tiên contour lớn nhất
    for contour in contours[:10]:  # Kiểm tra 10 contour lớn nhất
        area = cv2.contourArea(contour)
        
        # Contour phải chiếm ít nhất 5% diện tích ảnh
        if area < 0.05 * image_area:
            continue
        
        # Xấp xỉ đa giác
        perimeter = cv2.arcLength(contour, closed=True)
        
        # Thử nhiều epsilon khác nhau
        for epsilon_factor in [0.01, 0.02, 0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(contour, epsilon_factor * perimeter, closed=True)
            
            if len(approx) == 4:
                corners = approx.reshape(4, 2).astype(np.float32)
                corners = _sap_xep_goc(corners)
                
                # Kiểm tra tỷ lệ khung hình
                w_top = np.linalg.norm(corners[0] - corners[1])
                w_bot = np.linalg.norm(corners[3] - corners[2])
                h_left = np.linalg.norm(corners[0] - corners[3])
                h_right = np.linalg.norm(corners[1] - corners[2])
                
                avg_w = (w_top + w_bot) / 2.0
                avg_h = (h_left + h_right) / 2.0
                
                if avg_w > 0 and avg_h > 0:
                    ratio = max(avg_w, avg_h) / min(avg_w, avg_h)
                    
                    # Tỷ lệ A4 hoặc Letter: 1.2 - 2.0
                    if 1.2 <= ratio <= 2.0:
                        return corners
        
        # Nếu không tìm được 4 góc, thử Convex Hull
        hull = cv2.convexHull(contour)
        hull_perimeter = cv2.arcLength(hull, closed=True)
        
        for epsilon_factor in np.linspace(0.01, 0.10, 10):
            approx_hull = cv2.approxPolyDP(hull, epsilon_factor * hull_perimeter, closed=True)
            
            if len(approx_hull) == 4:
                corners = approx_hull.reshape(4, 2).astype(np.float32)
                corners = _sap_xep_goc(corners)
                
                # Kiểm tra tỷ lệ
                w_top = np.linalg.norm(corners[0] - corners[1])
                w_bot = np.linalg.norm(corners[3] - corners[2])
                h_left = np.linalg.norm(corners[0] - corners[3])
                h_right = np.linalg.norm(corners[1] - corners[2])
                
                avg_w = (w_top + w_bot) / 2.0
                avg_h = (h_left + h_right) / 2.0
                
                if avg_w > 0 and avg_h > 0:
                    ratio = max(avg_w, avg_h) / min(avg_w, avg_h)
                    
                    if 1.2 <= ratio <= 2.0:
                        return corners
    
    # Bước 6 - Nếu không tìm thấy
    raise ValueError(
        "Không tìm thấy tờ giấy thi trong ảnh. "
        "Vui lòng kiểm tra:\n"
        "  - Tờ giấy có nằm hoàn toàn trong khung ảnh không?\n"
        "  - Độ tương phản giữa giấy và nền có đủ rõ không?\n"
        "  - Ảnh có bị mờ hoặc nhiễu quá nhiều không?"
    )


def _sap_xep_goc(corners: np.ndarray) -> np.ndarray:
    """
    Sắp xếp 4 góc theo thứ tự: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    
    s = corners.sum(axis=1)
    diff = np.diff(corners, axis=1)
    
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]
    
    return rect


def nan_chinh_anh(anh: np.ndarray, 
                  cac_goc: np.ndarray,
                  chieu_rong: int = 800,
                  chieu_cao: int = 1200) -> np.ndarray:
    """
    Áp dụng phép biến đổi phối cảnh để nắn chỉnh ảnh.
    """
    dst_points = np.array([
        [0, 0],
        [chieu_rong - 1, 0],
        [chieu_rong - 1, chieu_cao - 1],
        [0, chieu_cao - 1]
    ], dtype=np.float32)
    
    matrix = cv2.getPerspectiveTransform(cac_goc, dst_points)
    warped = cv2.warpPerspective(anh, matrix, (chieu_rong, chieu_cao))
    
    return warped
