"""
Test cases cho module grader.py

Thành viên phụ trách: Thành viên 4
Nhiệm vụ: Test trích xuất ROI, phân đoạn, chấm điểm (dùng mock data)
"""

import pytest
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.grader import extract_bubble_grid, segment_bubbles, calculate_score


class TestExtractBubbleGrid:
    """Test cases cho hàm extract_bubble_grid()"""
    
    def test_extract_roi_basic(self):
        """Test cắt ROI cơ bản"""
        # TODO: Bước 1 - Tạo ảnh test 1200x800
        # TODO: Bước 2 - Gọi extract_bubble_grid() với ROI hợp lệ
        # TODO: Bước 3 - Assert output shape == (roi_height, roi_width)
        pass
    
    def test_extract_roi_out_of_bounds(self):
        """Test với ROI vượt quá giới hạn ảnh"""
        # TODO: Gọi extract_bubble_grid() với ROI lớn hơn ảnh
        # TODO: Phải xử lý lỗi hoặc clip về giới hạn hợp lệ
        pass


class TestSegmentBubbles:
    """Test cases cho hàm segment_bubbles()"""
    
    def test_adaptive_threshold(self):
        """Test phân ngưỡng adaptive"""
        # TODO: Bước 1 - Tạo ảnh xám test với vùng sáng/tối
        # TODO: Bước 2 - Gọi segment_bubbles(image, threshold_method="adaptive")
        # TODO: Bước 3 - Assert output là ảnh nhị phân (chỉ có 0 và 255)
        # TODO: Bước 4 - Assert output shape == input shape
        pass
    
    def test_otsu_threshold(self):
        """Test phân ngưỡng Otsu"""
        # TODO: Tương tự test_adaptive_threshold nhưng với method="otsu"
        pass
    
    def test_binary_threshold(self):
        """Test phân ngưỡng binary"""
        # TODO: Tương tự test_adaptive_threshold nhưng với method="binary"
        pass
    
    def test_invalid_threshold_method(self):
        """Test với method không hợp lệ"""
        # TODO: Sử dụng pytest.raises(ValueError)
        pass


class TestCalculateScore:
    """Test cases cho hàm calculate_score() - DÙNG MOCK DATA"""
    
    def test_calculate_score_all_correct(self):
        """Test chấm điểm khi tất cả đáp án đúng"""
        # TODO: Bước 1 - Tạo ảnh nhị phân MOCK với 4 câu, mỗi câu 4 lựa chọn
        #                Ví dụ: Câu 1 chọn A, Câu 2 chọn C, Câu 3 chọn B, Câu 4 chọn D
        # TODO: Bước 2 - Tạo answer_key giống với đáp án trong ảnh mock
        # TODO: Bước 3 - Gọi calculate_score()
        # TODO: Bước 4 - Assert correct_count == 4
        # TODO: Bước 5 - Assert score == 10.0
        pass
    
    def test_calculate_score_partial_correct(self):
        """Test chấm điểm khi một số đáp án đúng"""
        # TODO: Tạo mock data với 2/4 câu đúng
        # TODO: Assert correct_count == 2
        # TODO: Assert score == 5.0
        pass
    
    def test_calculate_score_all_wrong(self):
        """Test chấm điểm khi tất cả sai"""
        # TODO: Tạo mock data với 0/4 câu đúng
        # TODO: Assert correct_count == 0
        # TODO: Assert score == 0.0
        pass
    
    def test_calculate_score_no_answer(self):
        """Test khi học sinh không tô ô nào"""
        # TODO: Tạo ảnh nhị phân toàn đen (không có pixel trắng)
        # TODO: Kiểm tra xử lý trường hợp này
        pass
    
    def test_calculate_score_multiple_answers(self):
        """Test khi học sinh tô nhiều ô trong 1 câu"""
        # TODO: Tạo mock data với 2 ô được tô trong cùng 1 câu
        # TODO: Câu này phải tính là sai
        pass


# Helper function để tạo mock data
def create_mock_bubble_image(answers, num_questions=4, choices_per_question=4):
    """
    Tạo ảnh nhị phân mock cho test.
    
    Args:
        answers (dict): {1: 'A', 2: 'C', 3: 'B', 4: 'D'}
        num_questions (int): Số câu hỏi
        choices_per_question (int): Số lựa chọn mỗi câu
    
    Returns:
        np.ndarray: Ảnh nhị phân mock
    """
    # TODO: Bước 1 - Tạo ảnh đen với kích thước phù hợp
    # TODO: Bước 2 - Tính kích thước mỗi bubble
    # TODO: Bước 3 - Lặp qua answers và vẽ vùng trắng cho các ô được chọn
    # TODO: Bước 4 - Return ảnh mock
    pass


# Chạy tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
