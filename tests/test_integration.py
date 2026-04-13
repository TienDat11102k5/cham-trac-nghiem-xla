"""
Test tích hợp toàn bộ pipeline (End-to-End Test)

Thành viên phụ trách: Thành viên 5
Nhiệm vụ: Test toàn bộ luồng từ đầu đến cuối
"""

import pytest
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import load_image, convert_to_grayscale, apply_noise_filter
from src.transform import detect_edges, find_document_corners, apply_perspective_transform
from src.grader import extract_bubble_grid, segment_bubbles, calculate_score


class TestPipelineIntegration:
    """Test tích hợp toàn bộ pipeline"""
    
    def test_full_pipeline_with_mock_image(self):
        """
        Test toàn bộ pipeline từ đầu đến cuối với ảnh mock.
        
        Pipeline:
        1. Tạo ảnh mock (giả lập ảnh bài thi thật)
        2. Tiền xử lý
        3. Phát hiện biên và tìm góc
        4. Nắn chỉnh
        5. Trích xuất ROI
        6. Phân đoạn
        7. Chấm điểm
        """
        # TODO: Bước 1 - Tạo ảnh mock bài thi (có thể dùng helper function)
        # TODO: Bước 2 - Lưu ảnh mock vào file tạm
        # TODO: Bước 3 - Chạy toàn bộ pipeline:
        #                - load_image()
        #                - convert_to_grayscale()
        #                - apply_noise_filter()
        #                - detect_edges()
        #                - find_document_corners()
        #                - apply_perspective_transform()
        #                - extract_bubble_grid()
        #                - segment_bubbles()
        #                - calculate_score()
        # TODO: Bước 4 - Assert kết quả cuối cùng đúng với đáp án đã biết
        # TODO: Bước 5 - Xóa file tạm
        pass
    
    def test_pipeline_with_real_image(self):
        """Test pipeline với ảnh thật (nếu có)"""
        # TODO: Kiểm tra xem có ảnh test thật trong data/raw/ không
        # TODO: Nếu có, chạy pipeline và kiểm tra kết quả
        # TODO: Nếu không có, skip test này
        pass
    
    def test_pipeline_error_handling(self):
        """Test xử lý lỗi trong pipeline"""
        # TODO: Test các trường hợp lỗi:
        #       - File không tồn tại
        #       - Không tìm thấy tờ giấy
        #       - ROI không hợp lệ
        pass


def create_mock_exam_sheet(answers, image_size=(1200, 800)):
    """
    Tạo ảnh mock bài thi trắc nghiệm hoàn chỉnh.
    
    Args:
        answers (dict): Đáp án đã chọn {1: 'A', 2: 'C', ...}
        image_size (tuple): Kích thước ảnh (height, width)
    
    Returns:
        np.ndarray: Ảnh mock bài thi
    """
    # TODO: Bước 1 - Tạo ảnh trắng
    # TODO: Bước 2 - Vẽ viền tờ giấy (hình chữ nhật đen)
    # TODO: Bước 3 - Vẽ lưới các ô trắc nghiệm
    # TODO: Bước 4 - Tô đen các ô theo answers
    # TODO: Bước 5 - Thêm nhiễu nhẹ để giống ảnh thật
    # TODO: Bước 6 - Return ảnh mock
    pass


# Chạy tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
