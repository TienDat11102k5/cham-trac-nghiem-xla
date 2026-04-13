"""
Test cases cho module preprocessing.py

Thành viên phụ trách: Thành viên 2
Nhiệm vụ: Test các hàm đọc ảnh, chuyển xám, khử nhiễu
"""

import pytest
import cv2
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import load_image, convert_to_grayscale, apply_noise_filter


class TestLoadImage:
    """Test cases cho hàm load_image()"""
    
    def test_load_valid_image(self):
        """Test đọc ảnh hợp lệ"""
        # TODO: Bước 1 - Tạo ảnh test bằng numpy
        # TODO: Bước 2 - Lưu ảnh test vào file tạm
        # TODO: Bước 3 - Gọi load_image() và kiểm tra kết quả
        # TODO: Bước 4 - Assert image is not None
        # TODO: Bước 5 - Assert image.shape có 3 dimensions (height, width, channels)
        # TODO: Bước 6 - Xóa file test
        pass
    
    def test_load_nonexistent_image(self):
        """Test đọc ảnh không tồn tại - phải raise FileNotFoundError"""
        # TODO: Bước 1 - Gọi load_image() với đường dẫn không tồn tại
        # TODO: Bước 2 - Sử dụng pytest.raises(FileNotFoundError)
        pass
    
    def test_load_invalid_file(self):
        """Test đọc file không phải ảnh - phải raise ValueError"""
        # TODO: Bước 1 - Tạo file text tạm
        # TODO: Bước 2 - Gọi load_image() với file text
        # TODO: Bước 3 - Sử dụng pytest.raises(ValueError)
        pass


class TestConvertToGrayscale:
    """Test cases cho hàm convert_to_grayscale()"""
    
    def test_convert_color_to_gray(self):
        """Test chuyển ảnh màu sang xám"""
        # TODO: Bước 1 - Tạo ảnh màu test (BGR) với shape (100, 100, 3)
        # TODO: Bước 2 - Gọi convert_to_grayscale()
        # TODO: Bước 3 - Assert output shape == (100, 100) (2D)
        # TODO: Bước 4 - Assert output dtype == np.uint8
        pass
    
    def test_convert_already_gray(self):
        """Test với ảnh đã là xám"""
        # TODO: Bước 1 - Tạo ảnh xám test với shape (100, 100)
        # TODO: Bước 2 - Gọi convert_to_grayscale()
        # TODO: Bước 3 - Kiểm tra xem hàm xử lý đúng không (có thể return nguyên hoặc báo lỗi)
        pass


class TestApplyNoiseFilter:
    """Test cases cho hàm apply_noise_filter()"""
    
    def test_gaussian_filter(self):
        """Test Gaussian blur"""
        # TODO: Bước 1 - Tạo ảnh xám test với nhiễu
        # TODO: Bước 2 - Gọi apply_noise_filter(image, filter_type="gaussian", kernel_size=5)
        # TODO: Bước 3 - Assert output shape == input shape
        # TODO: Bước 4 - Assert output dtype == np.uint8
        pass
    
    def test_median_filter(self):
        """Test Median blur"""
        # TODO: Tương tự test_gaussian_filter nhưng với filter_type="median"
        pass
    
    def test_bilateral_filter(self):
        """Test Bilateral filter"""
        # TODO: Tương tự test_gaussian_filter nhưng với filter_type="bilateral"
        pass
    
    def test_invalid_filter_type(self):
        """Test với filter_type không hợp lệ - phải raise ValueError"""
        # TODO: Bước 1 - Tạo ảnh test
        # TODO: Bước 2 - Gọi apply_noise_filter() với filter_type="invalid"
        # TODO: Bước 3 - Sử dụng pytest.raises(ValueError)
        pass
    
    def test_invalid_kernel_size(self):
        """Test với kernel_size chẵn - phải raise ValueError"""
        # TODO: Bước 1 - Tạo ảnh test
        # TODO: Bước 2 - Gọi apply_noise_filter() với kernel_size=4 (số chẵn)
        # TODO: Bước 3 - Sử dụng pytest.raises(ValueError)
        pass


# Chạy tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
