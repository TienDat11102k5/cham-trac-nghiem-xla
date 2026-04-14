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

from src.preprocessing import doc_anh, chuyen_xam, loc_nhieu


class TestDocAnh:
    """Test cases cho hàm doc_anh()"""

    def test_doc_anh_hop_le(self, tmp_path):
        """Test đọc ảnh hợp lệ — ảnh được tạo bằng numpy, lưu tạm, đọc lại"""
        # Bước 1 - Tạo ảnh mock BGR (50×50, 3 channels)
        anh_test = np.zeros((50, 50, 3), dtype=np.uint8)
        anh_test[10:40, 10:40] = [255, 128, 0]  # vùng màu cam

        # Bước 2 - Lưu ảnh test vào file tạm (pytest tự dọn sau khi test xong)
        duong_dan = str(tmp_path / "anh_test.jpg")
        cv2.imwrite(duong_dan, anh_test)

        # Bước 3 - Gọi doc_anh() và kiểm tra kết quả
        ket_qua = doc_anh(duong_dan)

        # Bước 4 - Assert kết quả hợp lệ
        assert ket_qua is not None
        assert ket_qua.ndim == 3, "Ảnh màu phải có 3 chiều (H, W, C)"
        assert ket_qua.shape[2] == 3, "Phải có 3 channel (BGR)"
        assert ket_qua.dtype == np.uint8, "dtype phải là uint8"

    def test_doc_anh_file_khong_ton_tai(self):
        """Test đọc ảnh không tồn tại - phải raise FileNotFoundError"""
        with pytest.raises(FileNotFoundError) as thong_tin_loi:
            doc_anh("duong/dan/khong/ton/tai/anh.jpg")

        # Kiểm tra thông báo lỗi có chứa đường dẫn
        assert "duong/dan/khong/ton/tai/anh.jpg" in str(thong_tin_loi.value)

    def test_doc_anh_file_khong_phai_anh(self, tmp_path):
        """Test đọc file không phải ảnh (file text) - phải raise ValueError"""
        # Bước 1 - Tạo file text giả dạng .jpg
        file_text = tmp_path / "file_gia_mao.jpg"
        file_text.write_text("Đây là file text, không phải ảnh!")

        # Bước 2 - doc_anh() phải nhận ra file bị hỏng / không phải ảnh
        with pytest.raises(ValueError) as thong_tin_loi:
            doc_anh(str(file_text))

        assert "file_gia_mao.jpg" in str(thong_tin_loi.value)


class TestChuyenXam:
    """Test cases cho hàm chuyen_xam()"""

    def test_chuyen_anh_mau_sang_xam(self):
        """Test chuyển ảnh màu BGR sang ảnh xám"""
        # Bước 1 - Tạo ảnh màu mock BGR với shape (100, 100, 3)
        anh_mau = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Bước 2 - Gọi chuyen_xam()
        anh_xam = chuyen_xam(anh_mau)

        # Bước 3 - Assert output shape phải là 2D (H, W) không có channel
        assert anh_xam.ndim == 2, "Ảnh xám phải là mảng 2D"
        assert anh_xam.shape == (100, 100), "Shape phải là (100, 100)"

        # Bước 4 - Assert dtype giữ nguyên uint8
        assert anh_xam.dtype == np.uint8

    def test_chuyen_anh_da_xam_return_truc_tiep(self):
        """Test với ảnh đã là xám (2D) — phải return trực tiếp (Quyết định A)"""
        # Bước 1 - Tạo ảnh xám mock với shape (100, 100) — đã là 2D
        anh_xam_goc = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

        # Bước 2 - Gọi chuyen_xam() với ảnh đã xám
        ket_qua = chuyen_xam(anh_xam_goc)

        # Bước 3 - Phải return đúng object (không crash, không thay đổi)
        assert ket_qua.ndim == 2
        assert ket_qua.shape == (100, 100)
        # Kiểm tra là cùng một object (return trực tiếp, không tạo bản sao)
        assert np.array_equal(ket_qua, anh_xam_goc)


class TestLocNhieu:
    """Test cases cho hàm loc_nhieu()"""

    @pytest.fixture
    def anh_xam_mau(self):
        """Fixture tạo ảnh xám mock có nhiễu dùng chung cho các test"""
        # Ảnh nền xám + thêm nhiễu ngẫu nhiên để có nội dung thực tế
        anh_nen = np.full((100, 100), 128, dtype=np.uint8)
        nhieu = np.random.randint(-30, 30, (100, 100), dtype=np.int16)
        anh_co_nhieu = np.clip(anh_nen.astype(np.int16) + nhieu, 0, 255).astype(np.uint8)
        return anh_co_nhieu

    def test_loc_nhieu_gaussian(self, anh_xam_mau):
        """Test Gaussian blur — bộ lọc mặc định cho ảnh scan"""
        ket_qua = loc_nhieu(anh_xam_mau, loai_loc="gaussian", kich_thuoc=5)

        assert ket_qua.shape == anh_xam_mau.shape, "Shape phải giữ nguyên sau lọc"
        assert ket_qua.dtype == np.uint8, "dtype phải giữ nguyên uint8"

    def test_loc_nhieu_median(self, anh_xam_mau):
        """Test Median blur — tốt cho nhiễu muối tiêu (salt-and-pepper)"""
        ket_qua = loc_nhieu(anh_xam_mau, loai_loc="median", kich_thuoc=5)

        assert ket_qua.shape == anh_xam_mau.shape, "Shape phải giữ nguyên sau lọc"
        assert ket_qua.dtype == np.uint8, "dtype phải giữ nguyên uint8"

    def test_loc_nhieu_bilateral(self, anh_xam_mau):
        """Test Bilateral filter — edge-preserving, giữ cạnh sắc nét"""
        ket_qua = loc_nhieu(anh_xam_mau, loai_loc="bilateral", kich_thuoc=5)

        assert ket_qua.shape == anh_xam_mau.shape, "Shape phải giữ nguyên sau lọc"
        assert ket_qua.dtype == np.uint8, "dtype phải giữ nguyên uint8"

    def test_loc_nhieu_loai_khong_hop_le(self, anh_xam_mau):
        """Test với loai_loc không hợp lệ — phải raise ValueError"""
        with pytest.raises(ValueError) as thong_tin_loi:
            loc_nhieu(anh_xam_mau, loai_loc="sobel")

        # Thông báo lỗi phải chứa tên bộ lọc không hợp lệ
        assert "sobel" in str(thong_tin_loi.value)

    def test_loc_nhieu_kich_thuoc_chan(self, anh_xam_mau):
        """Test với kich_thuoc chẵn (4) — phải raise ValueError"""
        with pytest.raises(ValueError) as thong_tin_loi:
            loc_nhieu(anh_xam_mau, loai_loc="gaussian", kich_thuoc=4)

        assert "4" in str(thong_tin_loi.value)


# Chạy tests trực tiếp
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
