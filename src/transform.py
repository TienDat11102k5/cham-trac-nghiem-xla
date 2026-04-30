import cv2
import numpy as np
from typing import Tuple, List, Optional

def tim_canh(anh_mo: np.ndarray,
             nguong_thap: int = 30,
             nguong_cao: int = 100) -> np.ndarray:
    canh = cv2.Canny(anh_mo, nguong_thap, nguong_cao)
    return canh


def tim_goc_giay(anh_canh: np.ndarray, auto_detect_cropped: bool = True) -> Optional[np.ndarray]:
    h_anh, w_anh = anh_canh.shape
    image_area = h_anh * w_anh

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    anh_dong = cv2.morphologyEx(anh_canh, cv2.MORPH_CLOSE, kernel, iterations=2)
    anh_dilate = cv2.dilate(anh_dong, kernel, iterations=1)
    
    contours, _ = cv2.findContours(anh_dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        if auto_detect_cropped:
            return None
        raise ValueError(
            "Không tìm thấy tờ giấy thi trong ảnh. "
            "Vui lòng kiểm tra:\n"
            "  - Tờ giấy có nằm hoàn toàn trong khung ảnh không?\n"
            "  - Độ tương phản giữa giấy và nền có đủ rõ không?\n"
            "  - Ảnh có bị mờ hoặc nhiễu quá nhiều không?"
        )
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Lấy contour lớn nhất
    largest_contour = contours[0]
    largest_area = cv2.contourArea(largest_contour)
    area_ratio = largest_area / image_area
    
    # Nếu contour lớn nhất chiếm < 30% diện tích -> không phải tờ giấy chính
    # Lưu ý: Do dùng RETR_LIST, contour đầu tiên có thể là toàn bộ ảnh (> 95%), ta sẽ xét trong vòng lặp.
    
    # Thử tìm 4 góc từ contour lớn nhất
    for contour in contours[:3]:
        area = cv2.contourArea(contour)
        area_ratio = area / image_area
        
        # Chỉ xét contour chiếm >= 30% và <= 95% diện tích ảnh
        # (Loại bỏ contour bao trọn toàn bộ ảnh do nhiễu lề)
        if area_ratio < 0.30 or area_ratio > 0.95:
            continue
        
        perimeter = cv2.arcLength(contour, closed=True)
        
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
                    
                    # Tỷ lệ A4: 1.2 - 2.0
                    if 1.2 <= ratio <= 2.0:
                        # Kiểm tra kích thước tối thiểu (tránh các khung nhỏ)
                        min_width = w_anh * 0.5
                        min_height = h_anh * 0.5
                        
                        if avg_w >= min_width and avg_h >= min_height:
                            return corners
        
        # Thử với convex hull
        hull = cv2.convexHull(contour)
        hull_perimeter = cv2.arcLength(hull, closed=True)
        
        for epsilon_factor in np.linspace(0.01, 0.10, 10):
            approx_hull = cv2.approxPolyDP(hull, epsilon_factor * hull_perimeter, closed=True)
            
            if len(approx_hull) == 4:
                corners = approx_hull.reshape(4, 2).astype(np.float32)
                corners = _sap_xep_goc(corners)
                
                w_top = np.linalg.norm(corners[0] - corners[1])
                w_bot = np.linalg.norm(corners[3] - corners[2])
                h_left = np.linalg.norm(corners[0] - corners[3])
                h_right = np.linalg.norm(corners[1] - corners[2])
                
                avg_w = (w_top + w_bot) / 2.0
                avg_h = (h_left + h_right) / 2.0
                
                if avg_w > 0 and avg_h > 0:
                    ratio = max(avg_w, avg_h) / min(avg_w, avg_h)
                    
                    if 1.2 <= ratio <= 2.0:
                        min_width = w_anh * 0.5
                        min_height = h_anh * 0.5
                        
                        if avg_w >= min_width and avg_h >= min_height:
                            return corners
    
    # Không tìm thấy contour phù hợp -> ảnh đã cắt sẵn
    if auto_detect_cropped:
        return None
    
    raise ValueError(
        "Không tìm thấy tờ giấy thi trong ảnh. "
        "Vui lòng kiểm tra:\n"
        "  - Tờ giấy có nằm hoàn toàn trong khung ảnh không?\n"
        "  - Độ tương phản giữa giấy và nền có đủ rõ không?\n"
        "  - Ảnh có bị mờ hoặc nhiễu quá nhiều không?"
    )


def _kiem_tra_anh_da_cat_san(corners: np.ndarray, w: int, h: int, margin: int = 20) -> bool:
    """
    Kiểm tra xem ảnh đã được cắt sẵn chưa bằng cách xem 4 góc có gần viền không.
    Args:
        corners: 4 góc của tờ giấy
        w: Chiều rộng ảnh
        h: Chiều cao ảnh
        margin: Khoảng cách tối đa từ góc đến viền để coi là đã cắt
    Returns:
        True nếu ảnh đã cắt sẵn
    """
    # Kiểm tra 4 góc có gần 4 góc ảnh không
    tl, tr, br, bl = corners
    
    near_top_left = (tl[0] < margin and tl[1] < margin)
    near_top_right = (tr[0] > w - margin and tr[1] < margin)
    near_bottom_right = (br[0] > w - margin and br[1] > h - margin)
    near_bottom_left = (bl[0] < margin and bl[1] > h - margin)
    
    # Nếu ít nhất 3/4 góc gần viền -> ảnh đã cắt sẵn
    count = sum([near_top_left, near_top_right, near_bottom_right, near_bottom_left])
    return count >= 3


def _sap_xep_goc(corners: np.ndarray) -> np.ndarray:
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
    dst_points = np.array([
        [0, 0],
        [chieu_rong - 1, 0],
        [chieu_rong - 1, chieu_cao - 1],
        [0, chieu_cao - 1]
    ], dtype=np.float32)
    
    matrix = cv2.getPerspectiveTransform(cac_goc, dst_points)
    warped = cv2.warpPerspective(anh, matrix, (chieu_rong, chieu_cao))
    
    return warped
