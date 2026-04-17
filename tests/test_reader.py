"""
Test cases cho module reader.py

Thành viên phụ trách: [Tên thành viên]
Nhiệm vụ: Test đọc mã đề, mã sinh viên
"""

import pytest
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reader import (
    extract_exam_code_region,
    read_exam_code,
    extract_student_id_region,
    read_student_id,
    visualize_exam_code_region,
    visualize_all_regions
)


class TestExtractExamCodeRegion:
    """Test cases cho hàm extract_exam_code_region()"""
    
    def test_extract_exam_code_region_basic(self):
        """Test cắt vùng mã đề cơ bản"""
        # Bước 1 - Tạo ảnh test 1200x800
        anh_test = np.ones((1200, 800), dtype=np.uint8) * 200
        
        # Bước 2 - Gọi extract_exam_code_region()
        region = extract_exam_code_region(anh_test, roi_x=600, roi_y=50, roi_width=150, roi_height=100)
        
        # Bước 3 - Assert output shape đúng
        assert region.shape == (100, 150), f"Shape không đúng: {region.shape}"
        assert region.dtype == np.uint8
    
    def test_extract_exam_code_region_out_of_bounds(self):
        """Test với ROI vượt quá giới hạn ảnh"""
        # Tạo ảnh nhỏ
        anh_test = np.ones((100, 100), dtype=np.uint8)
        
        # Gọi với ROI lớn hơn ảnh - phải raise ValueError
        with pytest.raises(ValueError) as exc_info:
            extract_exam_code_region(anh_test, roi_x=50, roi_y=50, roi_width=200, roi_height=200)
        
        assert "vượt quá giới hạn" in str(exc_info.value)


class TestReadExamCode:
    """Test cases cho hàm read_exam_code() - DÙNG MOCK DATA"""
    
    def test_read_exam_code_101(self):
        """Test đọc mã đề 101"""
        # Bước 1 - Tạo ảnh mock với mã đề 101
        mock_image = create_mock_exam_code_image("101")
        
        # Bước 2 - Gọi read_exam_code()
        exam_code = read_exam_code(mock_image, num_digits=3, threshold_method="binary")
        
        # Bước 3 - Assert exam_code == "101"
        assert exam_code == "101", f"Mã đề không đúng: {exam_code}"
    
    def test_read_exam_code_102(self):
        """Test đọc mã đề 102"""
        mock_image = create_mock_exam_code_image("102")
        exam_code = read_exam_code(mock_image, num_digits=3, threshold_method="binary")
        assert exam_code == "102", f"Mã đề không đúng: {exam_code}"
    
    def test_read_exam_code_234(self):
        """Test đọc mã đề 234"""
        mock_image = create_mock_exam_code_image("234")
        exam_code = read_exam_code(mock_image, num_digits=3, threshold_method="binary")
        assert exam_code == "234", f"Mã đề không đúng: {exam_code}"
    
    def test_read_exam_code_invalid_no_mark(self):
        """Test khi không có ô nào được tô"""
        # Tạo ảnh toàn TRẮNG (không có ô đen nào = không tô)
        mock_image = np.ones((100, 150), dtype=np.uint8) * 255
        
        # Phải raise ValueError
        with pytest.raises(ValueError) as exc_info:
            read_exam_code(mock_image, num_digits=3, threshold_method="binary")
        
        assert "không có ô nào được tô" in str(exc_info.value)
    
    def test_read_exam_code_4_digits(self):
        """Test đọc mã đề 4 chữ số"""
        mock_image = create_mock_exam_code_image("1024", num_digits=4, bubble_width=37)
        exam_code = read_exam_code(mock_image, num_digits=4, threshold_method="binary")
        assert exam_code == "1024", f"Mã đề không đúng: {exam_code}"


class TestReadStudentID:
    """Test cases cho hàm read_student_id() - TÙY CHỌN"""
    
    def test_read_student_id_basic(self):
        """Test đọc mã sinh viên cơ bản"""
        # Tạo mock MSSV 8 chữ số: 12345678
        mock_image = create_mock_exam_code_image("12345678", num_digits=8, bubble_width=25)
        student_id = read_student_id(mock_image, num_digits=8)
        assert student_id == "12345678", f"MSSV không đúng: {student_id}"


class TestVisualizeExamCodeRegion:
    """Test cases cho hàm visualize_exam_code_region()"""
    
    def test_visualize_exam_code_region(self):
        """Test vẽ khung ROI lên ảnh"""
        # Bước 1 - Tạo ảnh test
        anh_test = np.ones((1200, 800), dtype=np.uint8) * 200
        
        # Bước 2 - Gọi visualize_exam_code_region()
        vis_image = visualize_exam_code_region(anh_test, roi_x=600, roi_y=50, roi_width=150, roi_height=100)
        
        # Bước 3 - Assert output có vẽ khung (kiểm tra pixel màu xanh lá)
        assert vis_image.ndim == 3, "Ảnh visualization phải là màu (3 channels)"
        assert vis_image.shape == (1200, 800, 3)
        
        # Kiểm tra có pixel màu xanh lá (0, 255, 0) tại vị trí khung
        # Pixel tại góc trên trái khung
        pixel = vis_image[50, 600]
        assert pixel[1] == 255, "Phải có màu xanh lá tại vị trí khung"


class TestProcessRealImages:
    """Test xử lý ảnh thật từ data/raw/"""
    
    THU_MUC_NGUON = Path("data/raw")
    THU_MUC_DICH = Path("data/processed")
    
    def test_co_anh_trong_thu_muc_raw(self):
        """Kiểm tra data/raw/ có ảnh"""
        danh_sach = list(self.THU_MUC_NGUON.glob("*.jpg"))
        assert len(danh_sach) > 0, "Không có ảnh trong data/raw/"
    
    def test_visualize_roi_tren_anh_that(self):
        """Test vẽ ROI lên ảnh thật để kiểm tra vị trí"""
        from src.preprocessing import doc_anh, chuyen_xam, loc_nhieu
        from src.transform import tim_canh, tim_goc_giay, nan_chinh_anh
        
        danh_sach_anh = list(self.THU_MUC_NGUON.glob("*.jpg"))
        if not danh_sach_anh:
            pytest.skip("Không có ảnh trong data/raw/")
        
        # Tạo thư mục processed
        self.THU_MUC_DICH.mkdir(parents=True, exist_ok=True)
        
        for duong_dan_anh in danh_sach_anh[:1]:  # Chỉ test ảnh đầu tiên
            ten_goc = duong_dan_anh.stem
            
            # Pipeline xử lý
            anh_goc = doc_anh(str(duong_dan_anh))
            anh_xam = chuyen_xam(anh_goc)
            anh_mo = loc_nhieu(anh_xam, loai_loc="gaussian", kich_thuoc=5)
            anh_canh = tim_canh(anh_mo, nguong_thap=50, nguong_cao=150)
            
            try:
                cac_goc = tim_goc_giay(anh_canh)
                if cac_goc is not None:
                    anh_thang = nan_chinh_anh(anh_goc, cac_goc, chieu_rong=800, chieu_cao=1200)
                else:
                    anh_thang = anh_goc
                
                # Vẽ tất cả các ROI với tọa độ đã đo
                anh_vis = visualize_all_regions(
                    anh_thang,
                    exam_code_roi=(409, 378, 159, 391),
                    student_id_roi=(147, 379, 229, 393),
                    answer_roi=(147, 814, 418, 350)
                )
                
                # Lưu ảnh visualization
                duong_dan_vis = self.THU_MUC_DICH / f"{ten_goc}_roi_visualization.jpg"
                thanh_cong = cv2.imwrite(str(duong_dan_vis), anh_vis)
                assert thanh_cong, f"Không lưu được {duong_dan_vis}"
                
                print(f"\n✓ Đã tạo: {duong_dan_vis}")
                print("  → Mở file này để kiểm tra vị trí ROI có đúng không")
                
            except ValueError as e:
                print(f"\n⚠️  Bỏ qua {ten_goc}: {e}")


# Helper function để tạo mock data
def create_mock_exam_code_image(exam_code: str, 
                                num_digits: int = 3,
                                choices_per_digit: int = 10,
                                bubble_width: int = 50,
                                bubble_height: int = 10) -> np.ndarray:
    """
    Tạo ảnh nhị phân mock cho mã đề.
    
    Args:
        exam_code: Mã đề dạng string, ví dụ: "101", "102"
        num_digits: Số chữ số
        choices_per_digit: Số lựa chọn mỗi chữ số (0-9)
        bubble_width: Chiều rộng mỗi bubble
        bubble_height: Chiều cao mỗi bubble
    
    Returns:
        Ảnh nhị phân mock với mã đề đã tô
    
    Examples:
        >>> img = create_mock_exam_code_image("101")
        >>> print(img.shape)
        (100, 150)  # 10 hàng x 3 cột, mỗi bubble 10x50
    """
    # Bước 1 - Tạo ảnh TRẮNG (255) - nền trắng như giấy thật
    height = choices_per_digit * bubble_height  # 10 * 10 = 100
    width = num_digits * bubble_width           # 3 * 50 = 150
    image = np.ones((height, width), dtype=np.uint8) * 255  # Nền trắng
    
    # Bước 2 - Lặp qua từng chữ số trong exam_code
    for digit_idx, digit_char in enumerate(exam_code):
        digit_value = int(digit_char)  # '1' -> 1
        
        # Tính vị trí bubble cần tô
        y1 = digit_value * bubble_height
        y2 = (digit_value + 1) * bubble_height
        x1 = digit_idx * bubble_width
        x2 = (digit_idx + 1) * bubble_width
        
        # Tô ĐEN bubble (giống bút chì tô trên giấy)
        image[y1:y2, x1:x2] = 0
    
    # Bước 3 - Return ảnh mock
    return image


# Chạy tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

