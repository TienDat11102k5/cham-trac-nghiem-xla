"""
Module chứa các hàm tiện ích cho hệ thống chấm trắc nghiệm tự động (OMR).

Module này cung cấp các hàm hỗ trợ như: hiển thị ảnh, lưu ảnh, đọc/ghi file JSON,
vẽ contours, và các tiện ích khác.
"""

import cv2
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path


def display_image(image: np.ndarray, 
                  window_name: str = "Image",
                  wait_key: int = 0) -> None:
    """
    Hiển thị ảnh trong cửa sổ OpenCV.
    
    Args:
        image (np.ndarray): Ảnh cần hiển thị.
        window_name (str, optional): Tên cửa sổ. Mặc định là "Image".
        wait_key (int, optional): Thời gian chờ (ms). 0 = chờ phím bất kỳ. Mặc định là 0.
    
    Returns:
        None
    
    Examples:
        >>> display_image(image, "Original Image", wait_key=0)
        >>> # Nhấn phím bất kỳ để đóng cửa sổ
    """
    # TODO: Bước 1 - Sử dụng cv2.imshow() để hiển thị ảnh
    # TODO: Bước 2 - Sử dụng cv2.waitKey(wait_key) để chờ
    # TODO: Bước 3 - Sử dụng cv2.destroyAllWindows() để đóng tất cả cửa sổ
    raise NotImplementedError


def save_image(image: np.ndarray, 
               output_path: str,
               create_dirs: bool = True) -> bool:
    """
    Lưu ảnh vào file.
    
    Args:
        image (np.ndarray): Ảnh cần lưu.
        output_path (str): Đường dẫn file đầu ra.
        create_dirs (bool, optional): Tự động tạo thư mục nếu chưa tồn tại. Mặc định là True.
    
    Returns:
        bool: True nếu lưu thành công, False nếu thất bại.
    
    Examples:
        >>> success = save_image(processed_image, "output/result.jpg")
        >>> if success:
        ...     print("Đã lưu ảnh thành công!")
    """
    # TODO: Bước 1 - Nếu create_dirs=True, tạo thư mục cha nếu chưa tồn tại
    #                Sử dụng Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # TODO: Bước 2 - Sử dụng cv2.imwrite(output_path, image) để lưu ảnh
    # TODO: Bước 3 - Return True nếu thành công, False nếu thất bại
    raise NotImplementedError


def load_answer_key_from_json(json_path: str) -> Dict[int, str]:
    """
    Đọc đáp án chuẩn từ file JSON.
    
    Args:
        json_path (str): Đường dẫn đến file JSON chứa đáp án.
    
    Returns:
        Dict[int, str]: Dictionary với key là số câu hỏi, value là đáp án.
    
    JSON Format:
        {
            "1": "A",
            "2": "C",
            "3": "B",
            ...
        }
    
    Examples:
        >>> answer_key = load_answer_key_from_json("data/answer_key.json")
        >>> print(answer_key[1])
        'A'
    """
    # TODO: Bước 1 - Mở file JSON bằng open() và json.load()
    # TODO: Bước 2 - Convert key từ string sang int: {int(k): v for k, v in data.items()}
    # TODO: Bước 3 - Return dictionary đã convert
    # TODO: Bước 4 - Xử lý exception nếu file không tồn tại hoặc format không hợp lệ
    raise NotImplementedError


def save_results_to_json(results: Dict,
                         output_path: str) -> bool:
    """
    Lưu kết quả chấm điểm vào file JSON.
    
    Args:
        results (Dict): Dictionary chứa kết quả chấm điểm.
        output_path (str): Đường dẫn file JSON đầu ra.
    
    Returns:
        bool: True nếu lưu thành công, False nếu thất bại.
    
    Results Format:
        {
            "image_path": "data/test_sheet_01.jpg",
            "num_questions": 40,
            "correct_count": 35,
            "score": 8.75,
            "student_answers": {"1": "A", "2": "C", ...},
            "answer_key": {"1": "A", "2": "C", ...}
        }
    
    Examples:
        >>> results = {"score": 8.5, "correct_count": 34}
        >>> save_results_to_json(results, "output/results.json")
    """
    # TODO: Bước 1 - Tạo thư mục cha nếu chưa tồn tại
    # TODO: Bước 2 - Mở file và sử dụng json.dump() với indent=4 để format đẹp
    # TODO: Bước 3 - Return True nếu thành công, False nếu thất bại
    raise NotImplementedError


def draw_contours(image: np.ndarray,
                 contours: List[np.ndarray],
                 color: Tuple[int, int, int] = (0, 255, 0),
                 thickness: int = 2) -> np.ndarray:
    """
    Vẽ contours lên ảnh (dùng để debug và visualize).
    
    Args:
        image (np.ndarray): Ảnh gốc (sẽ được copy để không thay đổi ảnh gốc).
        contours (List[np.ndarray]): Danh sách các contours cần vẽ.
        color (Tuple[int, int, int], optional): Màu BGR. Mặc định là xanh lá (0, 255, 0).
        thickness (int, optional): Độ dày đường vẽ. Mặc định là 2.
    
    Returns:
        np.ndarray: Ảnh đã vẽ contours.
    
    Examples:
        >>> contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        >>> image_with_contours = draw_contours(image, contours, color=(0, 255, 0))
    """
    # TODO: Bước 1 - Copy ảnh gốc: output = image.copy()
    # TODO: Bước 2 - Nếu ảnh là grayscale, convert sang BGR để vẽ màu:
    #                if len(output.shape) == 2:
    #                    output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    # TODO: Bước 3 - Sử dụng cv2.drawContours() để vẽ tất cả contours
    # TODO: Bước 4 - Return ảnh đã vẽ
    raise NotImplementedError


def draw_corners(image: np.ndarray,
                corners: np.ndarray,
                radius: int = 10,
                color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
    """
    Vẽ các điểm góc lên ảnh (dùng để debug).
    
    Args:
        image (np.ndarray): Ảnh gốc.
        corners (np.ndarray): Mảng 4 góc với shape (4, 2).
        radius (int, optional): Bán kính vòng tròn đánh dấu góc. Mặc định là 10.
        color (Tuple[int, int, int], optional): Màu BGR. Mặc định là đỏ (0, 0, 255).
    
    Returns:
        np.ndarray: Ảnh đã vẽ các góc.
    
    Examples:
        >>> image_with_corners = draw_corners(image, corners, radius=10)
    """
    # TODO: Bước 1 - Copy ảnh gốc
    # TODO: Bước 2 - Nếu ảnh là grayscale, convert sang BGR
    # TODO: Bước 3 - Lặp qua từng góc và vẽ vòng tròn:
    #                for i, corner in enumerate(corners):
    #                    cv2.circle(output, tuple(corner.astype(int)), radius, color, -1)
    #                    cv2.putText(output, str(i), tuple(corner.astype(int)), 
    #                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    # TODO: Bước 4 - Return ảnh đã vẽ
    raise NotImplementedError


def order_points(points: np.ndarray) -> np.ndarray:
    """
    Sắp xếp 4 điểm theo thứ tự: top-left, top-right, bottom-right, bottom-left.
    
    Args:
        points (np.ndarray): Mảng 4 điểm với shape (4, 2).
    
    Returns:
        np.ndarray: Mảng 4 điểm đã được sắp xếp, shape (4, 2).
    
    Notes:
        - Top-left: điểm có tổng (x+y) nhỏ nhất
        - Top-right: điểm có hiệu (y-x) nhỏ nhất
        - Bottom-right: điểm có tổng (x+y) lớn nhất
        - Bottom-left: điểm có hiệu (y-x) lớn nhất
    
    Examples:
        >>> unordered = np.array([[100, 200], [500, 150], [520, 800], [80, 820]])
        >>> ordered = order_points(unordered)
    """
    # TODO: Bước 1 - Khởi tạo mảng kết quả: rect = np.zeros((4, 2), dtype=np.float32)
    # TODO: Bước 2 - Tính tổng và hiệu:
    #                s = points.sum(axis=1)  # x + y
    #                diff = np.diff(points, axis=1)  # y - x
    # TODO: Bước 3 - Xác định các góc:
    #                rect[0] = points[np.argmin(s)]  # top-left
    #                rect[2] = points[np.argmax(s)]  # bottom-right
    #                rect[1] = points[np.argmin(diff)]  # top-right
    #                rect[3] = points[np.argmax(diff)]  # bottom-left
    # TODO: Bước 4 - Return mảng đã sắp xếp
    raise NotImplementedError


def create_output_directories() -> None:
    """
    Tạo các thư mục cần thiết cho output nếu chưa tồn tại.
    
    Thư mục được tạo:
        - output/
        - output/images/
        - output/results/
        - data/processed/
    
    Returns:
        None
    """
    # TODO: Bước 1 - Định nghĩa danh sách các thư mục cần tạo
    # TODO: Bước 2 - Lặp qua từng thư mục và tạo bằng Path().mkdir(parents=True, exist_ok=True)
    # TODO: Bước 3 - In thông báo đã tạo thư mục thành công
    raise NotImplementedError
