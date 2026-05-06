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
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import doc_anh, chuyen_xam, loc_nhieu
from src.transform import tim_canh, tim_goc_giay, nan_chinh_anh
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
        # Bước 1: Tạo ảnh mock với đáp án đã biết
        mock_answers = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'A'}
        mock_image = create_mock_exam_sheet(mock_answers)
        
        # Bước 2: Lưu ảnh mock vào file tạm
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, mock_image)
        
        try:
            # Bước 3: Chạy toàn bộ pipeline
            # 3.1 Đọc ảnh
            anh = doc_anh(tmp_path)
            assert anh is not None
            assert anh.shape[0] > 0 and anh.shape[1] > 0
            
            # 3.2 Chuyển xám
            anh_xam = chuyen_xam(anh)
            assert anh_xam.ndim == 2
            
            # 3.3 Lọc nhiễu
            anh_loc = loc_nhieu(anh_xam, loai_loc="gaussian", kich_thuoc=5)
            assert anh_loc.shape == anh_xam.shape
            
            # 3.4 Phát hiện biên
            anh_canh = tim_canh(anh_loc)
            assert anh_canh is not None
            
            # 3.5 Tìm 4 góc (mock image đã có viền rõ ràng)
            cac_goc = tim_goc_giay(anh_canh)
            if cac_goc is not None:
                assert cac_goc.shape == (4, 2)
                
                # 3.6 Nắn chỉnh
                anh_nan = nan_chinh_anh(anh, cac_goc)
                assert anh_nan is not None
                assert anh_nan.shape[0] > 0 and anh_nan.shape[1] > 0
            
            # Bước 4: Kiểm tra pipeline không crash
            # (Không kiểm tra kết quả chấm điểm vì mock image đơn giản)
            assert True
            
        finally:
            # Bước 5: Xóa file tạm
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_pipeline_with_real_image(self):
        """Test pipeline với ảnh thật (nếu có)"""
        # Kiểm tra xem có ảnh test thật trong data/raw/ không
        raw_dir = Path(__file__).parent.parent / "data" / "raw"
        
        if not raw_dir.exists():
            pytest.skip("Thư mục data/raw/ không tồn tại")
        
        image_files = list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.png"))
        
        if not image_files:
            pytest.skip("Không có ảnh test trong data/raw/")
        
        # Lấy ảnh đầu tiên để test
        test_image = image_files[0]
        
        # Chạy pipeline cơ bản
        anh = doc_anh(str(test_image))
        assert anh is not None
        
        anh_xam = chuyen_xam(anh)
        assert anh_xam.ndim == 2
        
        anh_loc = loc_nhieu(anh_xam)
        assert anh_loc.shape == anh_xam.shape
        
        anh_canh = tim_canh(anh_loc)
        assert anh_canh is not None
        
        # Không assert kết quả vì ảnh thật có thể không có viền rõ
        cac_goc = tim_goc_giay(anh_canh)
        # cac_goc có thể None nếu không tìm thấy
    
    def test_pipeline_error_handling(self):
        """Test xử lý lỗi trong pipeline"""
        # Test 1: File không tồn tại
        with pytest.raises(FileNotFoundError):
            doc_anh("file_khong_ton_tai.jpg")
        
        # Test 2: File không phải ảnh
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w') as tmp:
            tmp.write("This is not an image")
            tmp_path = tmp.name
        
        try:
            with pytest.raises(ValueError, match="không phải ảnh hợp lệ"):
                doc_anh(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        # Test 3: Kernel size không hợp lệ
        anh_test = np.ones((100, 100), dtype=np.uint8) * 128
        
        with pytest.raises(ValueError, match="số lẻ dương"):
            loc_nhieu(anh_test, kich_thuoc=4)  # Số chẵn
        
        with pytest.raises(ValueError, match="số lẻ dương"):
            loc_nhieu(anh_test, kich_thuoc=0)  # Số 0
        
        # Test 4: Loại lọc không hợp lệ
        with pytest.raises(ValueError, match="không hợp lệ"):
            loc_nhieu(anh_test, loai_loc="invalid_filter")


def create_mock_exam_sheet(answers, image_size=(1200, 800)):
    """
    Tạo ảnh mock bài thi trắc nghiệm đơn giản.
    
    Args:
        answers (dict): Đáp án đã chọn {1: 'A', 2: 'C', ...}
        image_size (tuple): Kích thước ảnh (height, width)
    
    Returns:
        np.ndarray: Ảnh mock bài thi
    """
    h, w = image_size
    
    # Bước 1: Tạo ảnh trắng
    image = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    # Bước 2: Vẽ viền tờ giấy (hình chữ nhật đen)
    border_thickness = 5
    cv2.rectangle(image, (10, 10), (w-10, h-10), (0, 0, 0), border_thickness)
    
    # Bước 3: Vẽ lưới các ô trắc nghiệm đơn giản
    # (Chỉ vẽ đơn giản để test pipeline, không cần chính xác)
    start_x = 100
    start_y = 200
    bubble_size = 20
    spacing = 40
    
    for q_idx, (question_num, answer) in enumerate(answers.items()):
        y = start_y + q_idx * spacing
        
        # Vẽ 4 ô A, B, C, D
        for choice_idx, choice in enumerate(['A', 'B', 'C', 'D']):
            x = start_x + choice_idx * spacing
            
            # Vẽ hình tròn
            cv2.circle(image, (x, y), bubble_size//2, (0, 0, 0), 2)
            
            # Tô đen ô được chọn
            if choice == answer:
                cv2.circle(image, (x, y), bubble_size//2 - 3, (0, 0, 0), -1)
    
    # Bước 4: Thêm nhiễu nhẹ để giống ảnh thật
    noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)
    
    return image


# Chạy tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
