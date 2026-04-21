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
        pytest.skip("Ham _cat_vung_roi da tu dong clip toa do, khong raise ValueError nua")


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
        mock_image = np.ones((100, 150), dtype=np.uint8) * 255

        with pytest.raises(ValueError) as exc_info:
            read_exam_code(mock_image, num_digits=3, threshold_method="binary")

        assert "Không phân biệt được ô nào được tô" in str(exc_info.value)
    
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
    
    def test_visualize_exam_code_region(self):
        pytest.skip("Ham visualize_exam_code_region da bi xoa")


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
        from src.reader import phat_hien_anchor, phan_loai_vung_roi, visualize_all_regions
        
        danh_sach_anh = list(self.THU_MUC_NGUON.glob("*.jpg"))
        if not danh_sach_anh:
            pytest.skip("Không có ảnh trong data/raw/")
        
        # Tạo thư mục processed
        self.THU_MUC_DICH.mkdir(parents=True, exist_ok=True)
        
        ket_qua = []
        
        for duong_dan_anh in danh_sach_anh:
            ten_goc = duong_dan_anh.stem
            
            try:
                # Bước 1: Đọc và tiền xử lý
                anh_goc = doc_anh(str(duong_dan_anh))
                anh_xam = chuyen_xam(anh_goc)
                anh_mo = loc_nhieu(anh_xam, loai_loc="gaussian", kich_thuoc=5)
                anh_canh = tim_canh(anh_mo, nguong_thap=50, nguong_cao=150)
                
                # Bước 2: Tìm góc và nắn chỉnh
                cac_goc = tim_goc_giay(anh_canh)
                if cac_goc is None:
                    print(f"Bỏ qua {ten_goc}: Không tìm thấy 4 góc")
                    continue
                
                anh_thang = nan_chinh_anh(anh_goc, cac_goc, chieu_rong=800, chieu_cao=1200)
                
                # Bước 3: Phát hiện anchor và phân loại ROI
                anchors = phat_hien_anchor(anh_thang)
                rois = phan_loai_vung_roi(anchors, *anh_thang.shape[:2])
                
                # Bước 4: Vẽ ROI lên ảnh bằng hàm có sẵn
                anh_ve = visualize_all_regions(anh_thang, rois)
                
                # Bước 5: Xuất ảnh ROI
                duong_dan_roi = self.THU_MUC_DICH / f"{ten_goc}_roi_regions.jpg"
                thanh_cong = cv2.imwrite(str(duong_dan_roi), anh_ve)
                assert thanh_cong, f"Không thể lưu ảnh {duong_dan_roi}"
                ket_qua.append(duong_dan_roi.name)
                
                # Bước 6: Đọc và xuất từng vùng riêng
                ma_de_region = extract_exam_code_region(anh_thang)
                duong_dan_ma_de = self.THU_MUC_DICH / f"{ten_goc}_ma_de.jpg"
                cv2.imwrite(str(duong_dan_ma_de), ma_de_region)
                ket_qua.append(duong_dan_ma_de.name)
                
                sbd_region = extract_student_id_region(anh_thang)
                duong_dan_sbd = self.THU_MUC_DICH / f"{ten_goc}_sbd.jpg"
                cv2.imwrite(str(duong_dan_sbd), sbd_region)
                ket_qua.append(duong_dan_sbd.name)
                
            except Exception as e:
                print(f"Lỗi xử lý {ten_goc}: {e}")
                continue
        
        assert len(ket_qua) > 0, "Không xuất được ảnh nào"
        print(f"\nĐã xuất {len(ket_qua)} ảnh vào {self.THU_MUC_DICH}:")
        for ten in ket_qua:
            print(f"  - {ten}")


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

