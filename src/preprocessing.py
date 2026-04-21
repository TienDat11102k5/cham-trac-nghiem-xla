import os
import cv2
import numpy as np
from typing import Optional


def doc_anh(duong_dan: str) -> np.ndarray:
    # cv2.imread() trả None thầm lặng cho cả 2 trường hợp:
    # không tồn tại VÀ không phải ảnh → cần phân biệt để raise đúng exception
    if not os.path.exists(duong_dan):
        raise FileNotFoundError(
            f"Không tìm thấy file ảnh: '{duong_dan}'"
        )
    anh = cv2.imread(duong_dan)
    if anh is None:
        raise ValueError(
            f"File không phải ảnh hợp lệ hoặc bị hỏng: '{duong_dan}'"
        )
    return anh

def chuyen_xam(anh: np.ndarray) -> np.ndarray:
    # Tránh crash khi hàm được gọi lại trong quá trình debug pipeline
    if anh.ndim == 2:
        return anh
    # OpenCV tối ưu nội bộ bằng SIMD, nhanh hơn tự tính bằng NumPy
    anh_xam = cv2.cvtColor(anh, cv2.COLOR_BGR2GRAY)
    return anh_xam

def loc_nhieu(anh_xam: np.ndarray,
              loai_loc: str = "gaussian",
              kich_thuoc: int = 5) -> np.ndarray:
    # Kernel cần có 1 pixel trung tâm chính xác → số chẵn không có tâm
    if kich_thuoc <= 0 or kich_thuoc % 2 == 0:
        raise ValueError(
            f"kich_thuoc phải là số lẻ dương, nhận được: {kich_thuoc}"
        )

    cac_bo_loc_hop_le = ("gaussian", "median", "bilateral")
    if loai_loc not in cac_bo_loc_hop_le:
        raise ValueError(
            f"loai_loc không hợp lệ: '{loai_loc}'. "
            f"Các giá trị hợp lệ: {cac_bo_loc_hop_le}"
        )
    # Phù hợp nhất cho nhiễu Gaussian — loại nhiễu phổ biến nhất trong ảnh scan
    if loai_loc == "gaussian":
        return cv2.GaussianBlur(anh_xam, (kich_thuoc, kich_thuoc), 0)
    # Loại bỏ được nhiễu muối tiêu (salt-and-pepper) vì median loại cực trị
    if loai_loc == "median":
        return cv2.medianBlur(anh_xam, kich_thuoc)
    # Edge-preserving: pixel có cường độ khác biệt lớn (= cạnh) không bị trộn
    return cv2.bilateralFilter(anh_xam, kich_thuoc, sigmaColor=75, sigmaSpace=75)
