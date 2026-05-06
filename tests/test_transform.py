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

from src.preprocessing import doc_anh, chuyen_xam, loc_nhieu
from src.transform import tim_canh, tim_goc_giay, nan_chinh_anh


class TestDetectEdges:
    """Test cases cho hàm tim_canh()"""
    
    def test_detect_edges_basic(self):
        """Test phát hiện biên cơ bản"""
        # Bước 1 - Tạo ảnh test với hình chữ nhật trắng trên nền đen
        anh_test = np.zeros((300, 400), dtype=np.uint8)
        cv2.rectangle(anh_test, (50, 50), (350, 250), 255, -1)
        
        # Bước 2 - Làm mờ ảnh bằng Gaussian
        anh_mo = cv2.GaussianBlur(anh_test, (5, 5), 0)
        
        # Bước 3 - Gọi tim_canh()
        canh = tim_canh(anh_mo, nguong_thap=50, nguong_cao=150)
        
        # Bước 4 - Assert output là ảnh nhị phân (chỉ có 0 và 255)
        assert canh.dtype == np.uint8
        assert set(np.unique(canh)).issubset({0, 255})
        
        # Bước 5 - Assert có phát hiện được biên (có pixel 255)
        assert np.any(canh == 255), "Không phát hiện được biên nào"
    
    def test_detect_edges_custom_thresholds(self):
        """Test với ngưỡng tùy chỉnh"""
        anh_test = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(anh_test, (30, 30), (170, 170), 255, -1)
        anh_mo = cv2.GaussianBlur(anh_test, (5, 5), 0)
        
        # Test với ngưỡng thấp
        canh_thap = tim_canh(anh_mo, nguong_thap=30, nguong_cao=90)
        assert np.any(canh_thap == 255)
        
        # Test với ngưỡng cao
        canh_cao = tim_canh(anh_mo, nguong_thap=100, nguong_cao=200)
        assert canh_cao.shape == anh_mo.shape


class TestFindDocumentCorners:
    """Test cases cho hàm tim_goc_giay()"""
    
    def test_find_corners_rectangle(self):
        """Test tìm 4 góc của hình chữ nhật"""
        # Bước 1 - Tạo ảnh test với hình chữ nhật rõ ràng
        anh_test = np.zeros((400, 500), dtype=np.uint8)
        cv2.rectangle(anh_test, (100, 80), (400, 320), 255, 2)
        
        # Bước 2 - Phát hiện biên
        anh_mo = cv2.GaussianBlur(anh_test, (5, 5), 0)
        canh = tim_canh(anh_mo)
        
        # Bước 3 - Gọi tim_goc_giay()
        cac_goc = tim_goc_giay(canh)
        
        # Bước 4 - Assert output shape == (4, 2)
        assert cac_goc.shape == (4, 2), f"Shape không đúng: {cac_goc.shape}"
        
        # Bước 5 - Assert dtype == np.float32
        assert cac_goc.dtype == np.float32, f"Dtype không đúng: {cac_goc.dtype}"
        
        # Bước 6 - Kiểm tra 4 góc có đúng thứ tự không
        # Top-left nên có tọa độ nhỏ nhất
        assert cac_goc[0][0] < cac_goc[1][0], "Top-left x phải nhỏ hơn top-right x"
        assert cac_goc[0][1] < cac_goc[3][1], "Top-left y phải nhỏ hơn bottom-left y"
    
    def test_find_corners_no_document(self):
        """Test với ảnh không có tờ giấy - phải raise ValueError"""
        # Bước 1 - Tạo ảnh trống hoàn toàn (không có contour)
        anh_trong = np.zeros((300, 300), dtype=np.uint8)
        
        # Bước 2 & 3 - Gọi tim_goc_giay() và expect ValueError
        with pytest.raises(ValueError, match="Không tìm thấy tờ giấy thi"):
            tim_goc_giay(anh_trong, auto_detect_cropped=False)
    
    def test_find_corners_multiple_contours(self):
        """Test với nhiều contours - phải chọn contour lớn nhất"""
        # Tạo ảnh với 2 hình chữ nhật, 1 lớn 1 nhỏ
        anh_test = np.zeros((500, 600), dtype=np.uint8)
        # Hình chữ nhật lớn
        cv2.rectangle(anh_test, (50, 50), (550, 450), 255, 2)
        # Hình chữ nhật nhỏ
        cv2.rectangle(anh_test, (200, 200), (300, 300), 255, 2)
        
        anh_mo = cv2.GaussianBlur(anh_test, (5, 5), 0)
        canh = tim_canh(anh_mo)
        cac_goc = tim_goc_giay(canh)
        
        # Kiểm tra đã chọn hình chữ nhật lớn (góc gần biên ảnh)
        assert cac_goc[0][0] < 100, "Nên chọn hình chữ nhật lớn"
        assert cac_goc.shape == (4, 2)


class TestApplyPerspectiveTransform:
    """Test cases cho hàm nan_chinh_anh()"""
    
    def test_perspective_transform_basic(self):
        """Test nắn chỉnh ảnh cơ bản"""
        # Bước 1 - Tạo ảnh test màu
        anh_test = np.ones((400, 500, 3), dtype=np.uint8) * 255
        cv2.rectangle(anh_test, (100, 100), (400, 300), (0, 0, 255), -1)
        
        # Bước 2 - Định nghĩa 4 góc nguồn (hình thang - giả lập góc nghiêng)
        cac_goc = np.array([
            [120, 80],   # top-left
            [380, 100],  # top-right
            [400, 320],  # bottom-right
            [100, 300]   # bottom-left
        ], dtype=np.float32)
        
        # Bước 3 - Gọi nan_chinh_anh()
        anh_thang = nan_chinh_anh(anh_test, cac_goc, chieu_rong=300, chieu_cao=400)
        
        # Bước 4 - Assert output shape đúng
        assert anh_thang.shape == (400, 300, 3), f"Shape không đúng: {anh_thang.shape}"
    
    def test_perspective_transform_custom_size(self):
        """Test với kích thước output tùy chỉnh"""
        anh_test = np.ones((300, 400, 3), dtype=np.uint8) * 128
        cac_goc = np.array([
            [50, 50],
            [350, 50],
            [350, 250],
            [50, 250]
        ], dtype=np.float32)
        
        # Test với kích thước khác nhau
        anh_thang_1 = nan_chinh_anh(anh_test, cac_goc, chieu_rong=600, chieu_cao=800)
        assert anh_thang_1.shape == (800, 600, 3)
        
        anh_thang_2 = nan_chinh_anh(anh_test, cac_goc, chieu_rong=200, chieu_cao=300)
        assert anh_thang_2.shape == (300, 200, 3)
    
    def test_perspective_transform_grayscale(self):
        """Test nắn chỉnh ảnh xám"""
        # Tạo ảnh xám (2D)
        anh_xam = np.ones((300, 400), dtype=np.uint8) * 200
        cv2.rectangle(anh_xam, (100, 100), (300, 200), 50, -1)
        
        cac_goc = np.array([
            [80, 80],
            [320, 80],
            [320, 220],
            [80, 220]
        ], dtype=np.float32)
        
        anh_thang = nan_chinh_anh(anh_xam, cac_goc, chieu_rong=400, chieu_cao=500)
        
        # Ảnh xám output vẫn là 2D
        assert anh_thang.shape == (500, 400), f"Shape không đúng cho ảnh xám: {anh_thang.shape}"


class TestProcessRealImages:
    """Test xử lý ảnh thật và tạo file processed (giống TV2)"""
    
    THU_MUC_NGUON = Path("data/raw")
    THU_MUC_DICH = Path("data/processed")
    
    def _lay_danh_sach_anh(self):
        """Lấy danh sách ảnh .jpg trong data/raw/"""
        if not self.THU_MUC_NGUON.exists():
            return []
        return list(self.THU_MUC_NGUON.glob("*.jpg"))
    
    def _ten_dau_ra(self, ten_goc: str, buoc: str) -> Path:
        """
        Tạo tên file đầu ra theo quy ước.
        
        Ví dụ: ten_goc='test_sheet_01', buoc='05_canh_canny'
        → data/processed/test_sheet_01_05_canh_canny.jpg
        """
        return self.THU_MUC_DICH / f"{ten_goc}_{buoc}.jpg"
    
    def _ve_goc_len_anh(self, anh: np.ndarray, cac_goc: np.ndarray) -> np.ndarray:
        """Vẽ 4 góc và viền lên ảnh để visualization."""
        anh_ve = anh.copy()
        
        # Chuyển sang màu nếu là ảnh xám
        if anh_ve.ndim == 2:
            anh_ve = cv2.cvtColor(anh_ve, cv2.COLOR_GRAY2BGR)
        
        # Vẽ viền nối 4 góc
        pts = cac_goc.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(anh_ve, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
        
        # Vẽ 4 góc với màu khác nhau
        mau_sac = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        nhan = ["TL", "TR", "BR", "BL"]
        
        for goc, mau, ten in zip(cac_goc, mau_sac, nhan):
            x, y = int(goc[0]), int(goc[1])
            cv2.circle(anh_ve, (x, y), 15, mau, -1)
            cv2.circle(anh_ve, (x, y), 15, (255, 255, 255), 2)
            cv2.putText(anh_ve, ten, (x - 20, y - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, mau, 2)
        
        return anh_ve
    
    def test_co_anh_trong_thu_muc_raw(self):
        """Kiểm tra data/raw/ tồn tại và có ít nhất 1 ảnh để test."""
        danh_sach = self._lay_danh_sach_anh()
        assert len(danh_sach) > 0, (
            f"Không tìm thấy ảnh nào trong '{self.THU_MUC_NGUON}'. "
            "Hãy thêm ít nhất 1 ảnh phiếu trắc nghiệm vào data/raw/"
        )
    
    def test_xu_ly_toan_bo_anh_trong_raw(self):
        """
        Test chính: Tự động xử lý TẤT CẢ ảnh trong data/raw/
        và lưu kết quả vào data/processed/ (giống TV2).
        """
        danh_sach_anh = self._lay_danh_sach_anh()
        
        if not danh_sach_anh:
            pytest.skip("Không có ảnh trong data/raw/ — bỏ qua test thực tế")
        
        # Tạo thư mục processed nếu chưa có
        self.THU_MUC_DICH.mkdir(parents=True, exist_ok=True)
        
        ket_qua_tong = []
        
        for duong_dan_anh in sorted(danh_sach_anh):
            ten_goc = duong_dan_anh.stem  # tên file không có đuôi
            
            # Bước 1 - Đọc ảnh gốc
            anh_goc = doc_anh(str(duong_dan_anh))
            assert anh_goc is not None and anh_goc.ndim == 3
            
            # Bước 2 - Tiền xử lý (TV2)
            anh_xam = chuyen_xam(anh_goc)
            anh_mo = loc_nhieu(anh_xam, loai_loc="gaussian", kich_thuoc=5)
            
            # Bước 3 - Phát hiện biên (TV3)
            anh_canh = tim_canh(anh_mo, nguong_thap=50, nguong_cao=150)
            duong_dan_canh = self._ten_dau_ra(ten_goc, "05_canh_canny")
            thanh_cong = cv2.imwrite(str(duong_dan_canh), anh_canh)
            assert thanh_cong
            ket_qua_tong.append(duong_dan_canh.name)
            
            # Bước 4 - Tìm 4 góc (TV3)
            try:
                cac_goc = tim_goc_giay(anh_canh, auto_detect_cropped=False)
                assert cac_goc.shape == (4, 2)
                
                # Vẽ góc lên ảnh gốc
                anh_ve_goc = self._ve_goc_len_anh(anh_goc, cac_goc)
                duong_dan_goc = self._ten_dau_ra(ten_goc, "06_tim_goc")
                thanh_cong = cv2.imwrite(str(duong_dan_goc), anh_ve_goc)
                assert thanh_cong
                ket_qua_tong.append(duong_dan_goc.name)
                
                # Bước 5 - Nắn chỉnh ảnh (TV3)
                anh_thang = nan_chinh_anh(anh_goc, cac_goc, chieu_rong=800, chieu_cao=1200)
                assert anh_thang.shape[:2] == (1200, 800)
                
                duong_dan_thang = self._ten_dau_ra(ten_goc, "07_nan_chinh")
                thanh_cong = cv2.imwrite(str(duong_dan_thang), anh_thang)
                assert thanh_cong
                ket_qua_tong.append(duong_dan_thang.name)
                
                # Bước 6 - Tạo ảnh so sánh
                h_goc, w_goc = anh_goc.shape[:2]
                ty_le = 400 / h_goc
                anh_goc_resize = cv2.resize(anh_goc, (int(w_goc * ty_le), 400))
                anh_thang_resize = cv2.resize(anh_thang, (int(800 * 400 / 1200), 400))
                anh_so_sanh = np.hstack([anh_goc_resize, anh_thang_resize])
                
                cv2.putText(anh_so_sanh, "TRUOC", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(anh_so_sanh, "SAU", (anh_goc_resize.shape[1] + 20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                duong_dan_so_sanh = self._ten_dau_ra(ten_goc, "08_so_sanh")
                thanh_cong = cv2.imwrite(str(duong_dan_so_sanh), anh_so_sanh)
                assert thanh_cong
                ket_qua_tong.append(duong_dan_so_sanh.name)
                
            except ValueError as e:
                # Nếu không tìm thấy tờ giấy, bỏ qua ảnh này
                print(f"\n⚠️  Bỏ qua {ten_goc}: {e}")
                continue
        
        # In báo cáo tóm tắt
        print(f"\n\n  📁 data/processed/ — {len(ket_qua_tong)} file đã tạo (TV3):")
        for ten_file in ket_qua_tong:
            print(f"     ✓ {ten_file}")


# Chạy tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
