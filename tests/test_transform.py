"""
Test cases cho module transform.py

Thành viên phụ trách: Thành viên 3
Nhiệm vụ: Test phát hiện biên, tìm góc, nắn chỉnh ảnh
"""

import pytest
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transform import detect_edges, find_document_corners, apply_perspective_transform


class TestDetectEdges:
    """Test cases cho hàm detect_edges()"""
    
    def test_detect_edges_basic(self):
        """Test phát hiện biên cơ bản"""
        # TODO: Bước 1 - Tạo ảnh test với hình chữ nhật trắng trên nền đen
        # TODO: Bước 2 - Làm mờ ảnh bằng Gaussian
        # TODO: Bước 3 - Gọi detect_edges()
        # TODO: Bước 4 - Assert output là ảnh nhị phân (chỉ có 0 và 255)
        # TODO: Bước 5 - Assert có phát hiện được biên (có pixel 255)
        pass
    
    def test_detect_edges_custom_thresholds(self):
        """Test với ngưỡng tùy chỉnh"""
        # TODO: Test với low_threshold và high_threshold khác nhau
        pass


class TestFindDocumentCorners:
    """Test cases cho hàm find_document_corners()"""
    
    def test_find_corners_rectangle(self):
        """Test tìm 4 góc của hình chữ nhật"""
        # TODO: Bước 1 - Tạo ảnh test với hình chữ nhật rõ ràng
        # TODO: Bước 2 - Phát hiện biên
        # TODO: Bước 3 - Gọi find_document_corners()
        # TODO: Bước 4 - Assert output shape == (4, 2)
        # TODO: Bước 5 - Assert dtype == np.float32
        # TODO: Bước 6 - Kiểm tra 4 góc có đúng thứ tự không (top-left, top-right, bottom-right, bottom-left)
        pass
    
    def test_find_corners_no_document(self):
        """Test với ảnh không có tờ giấy - phải raise ValueError"""
        # TODO: Bước 1 - Tạo ảnh trống hoặc nhiễu
        # TODO: Bước 2 - Gọi find_document_corners()
        # TODO: Bước 3 - Sử dụng pytest.raises(ValueError)
        pass
    
    def test_find_corners_multiple_contours(self):
        """Test với nhiều contours - phải chọn contour lớn nhất"""
        # TODO: Tạo ảnh với nhiều hình chữ nhật, kiểm tra chọn đúng cái lớn nhất
        pass


class TestApplyPerspectiveTransform:
    """Test cases cho hàm apply_perspective_transform()"""
    
    def test_perspective_transform_basic(self):
        """Test nắn chỉnh ảnh cơ bản"""
        # TODO: Bước 1 - Tạo ảnh test
        # TODO: Bước 2 - Định nghĩa 4 góc nguồn (có thể nghiêng)
        # TODO: Bước 3 - Gọi apply_perspective_transform()
        # TODO: Bước 4 - Assert output shape == (output_height, output_width, channels)
        pass
    
    def test_perspective_transform_custom_size(self):
        """Test với kích thước output tùy chỉnh"""
        # TODO: Test với output_width và output_height khác nhau
        pass
    
    def test_perspective_transform_grayscale(self):
        """Test nắn chỉnh ảnh xám"""
        # TODO: Test với ảnh xám (2D) thay vì ảnh màu (3D)
        pass


# Chạy tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
