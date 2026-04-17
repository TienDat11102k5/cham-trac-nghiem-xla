"""
Hệ thống chấm trắc nghiệm tự động (OMR) - Optical Mark Recognition.

File chính để chạy pipeline xử lý: Đọc ảnh -> Tiền xử lý -> Nắn chỉnh -> Chấm điểm.

Author: Computer Vision Team
Date: 2026
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import tkinter as tk
from tkinter import filedialog

from src.preprocessing import doc_anh, chuyen_xam, loc_nhieu
from src.transform import tim_canh, tim_goc_giay, nan_chinh_anh
from src.reader import extract_exam_code_region, read_exam_code, extract_student_id_region, read_student_id, visualize_all_regions
from src.config import SAMPLE_ANSWER_KEY


def doc_dap_an(answer_key_path: str) -> Dict[int, str]:
    """
    Đọc đáp án chuẩn từ file JSON.
    
    Args:
        answer_key_path: Đường dẫn đến file JSON chứa đáp án
        
    Returns:
        Dictionary với key là số câu (int), value là đáp án (str)
    
    Format JSON hỗ trợ:
        1. Format đơn giản: {"1": "A", "2": "C", ...}
        2. Format đầy đủ: {"answers": {"1": "A", "2": "C", ...}, "exam_id": "..."}
    """
    with open(answer_key_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Nếu có key "answers", lấy từ đó (format đầy đủ)
    if "answers" in data:
        answers = data["answers"]
    else:
        # Format đơn giản, toàn bộ file là đáp án
        answers = data
    
    return {int(k): v for k, v in answers.items()}


def luu_anh_trung_gian(anh: any, ten_file: str, thu_muc: str = "output") -> None:
    """
    Lưu ảnh trung gian vào thư mục output.
    
    Args:
        anh: Ảnh cần lưu
        ten_file: Tên file (bao gồm đuôi .jpg)
        thu_muc: Thư mục đích
    """
    Path(thu_muc).mkdir(parents=True, exist_ok=True)
    duong_dan = Path(thu_muc) / ten_file
    cv2.imwrite(str(duong_dan), anh)


def main(image_path: str, answer_key_path: Optional[str] = None, save_images: bool = True) -> None:
    """
    Hàm chính để chạy pipeline chấm trắc nghiệm tự động.
    
    Pipeline gồm các bước:
    1. Đọc ảnh từ file
    2. Tiền xử lý: Chuyển sang xám và khử nhiễu
    3. Phát hiện biên và tìm 4 góc tờ giấy thi
    4. Nắn chỉnh ảnh về mặt phẳng chuẩn
    5. Đọc mã đề thi và mã sinh viên
    6. [TODO] Trích xuất vùng đáp án và phân đoạn
    7. [TODO] Chấm điểm và in kết quả
    
    Args:
        image_path: Đường dẫn đến file ảnh bài thi cần chấm
        answer_key_path: Đường dẫn đến file đáp án chuẩn (JSON). Nếu None, dùng đáp án mẫu
        save_images: Có lưu ảnh trung gian vào thư mục output không
    """
    print("=" * 60)
    print("HỆ THỐNG CHẤM TRẮC NGHIỆM TỰ ĐỘNG (OMR)")
    print("=" * 60)
    print(f"Đang xử lý: {image_path}")
    print()
    
    try:
        # Bước 1 - ĐỌC ẢNH
        print("[1/8] Đọc ảnh...", end=" ")
        anh_goc = doc_anh(image_path)
        print("✓")
        
        # Bước 2 - TIỀN XỬ LÝ: Chuyển xám
        print("[2/8] Chuyển sang ảnh xám...", end=" ")
        anh_xam = chuyen_xam(anh_goc)
        if save_images:
            luu_anh_trung_gian(anh_xam, "01_anh_xam.jpg")
        print("✓")
        
        # Bước 3 - TIỀN XỬ LÝ: Khử nhiễu
        print("[3/8] Áp dụng bộ lọc khử nhiễu (Gaussian)...", end=" ")
        anh_mo = loc_nhieu(anh_xam, loai_loc="gaussian", kich_thuoc=5)
        if save_images:
            luu_anh_trung_gian(anh_mo, "02_loc_nhieu.jpg")
        print("✓")
        
        # Bước 4 - PHÁT HIỆN BIÊN
        print("[4/8] Phát hiện biên (Canny)...", end=" ")
        anh_canh = tim_canh(anh_mo, nguong_thap=50, nguong_cao=150)
        if save_images:
            luu_anh_trung_gian(anh_canh, "03_canh.jpg")
        print("✓")
        
        # Bước 5 - TÌM 4 GÓC TỜ GIẤY
        print("[5/8] Tìm 4 góc tờ giấy thi...", end=" ")
        cac_goc = tim_goc_giay(anh_canh, auto_detect_cropped=True)
        
        if cac_goc is None:
            print("⚠️  (Ảnh đã cắt sẵn)")
            # Nếu ảnh đã cắt sẵn, resize trực tiếp về 800x1200
            print("      → Resize ảnh về 800x1200...", end=" ")
            h_goc, w_goc = anh_goc.shape[:2]
            anh_nan_chinh = cv2.resize(anh_goc, (800, 1200))
            print(f"✓ (từ {w_goc}x{h_goc})")
            if save_images:
                luu_anh_trung_gian(anh_nan_chinh, "05_resize_800x1200.jpg")
        else:
            print("✓")
            
            if save_images:
                anh_ve_goc = anh_goc.copy()
                for i, goc in enumerate(cac_goc):
                    cv2.circle(anh_ve_goc, tuple(goc.astype(int)), 10, (0, 255, 0), -1)
                    cv2.putText(anh_ve_goc, str(i), tuple(goc.astype(int)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                luu_anh_trung_gian(anh_ve_goc, "04_tim_goc.jpg")
            
            # Bước 6 - NẮN CHỈNH ẢNH
            print("[6/8] Nắn chỉnh ảnh (Perspective Transform)...", end=" ")
            anh_nan_chinh = nan_chinh_anh(anh_goc, cac_goc, chieu_rong=800, chieu_cao=1200)
            if save_images:
                luu_anh_trung_gian(anh_nan_chinh, "05_nan_chinh.jpg")
            print("✓")
        
        # Bước 7 - ĐỌC MÃ ĐỀ VÀ MÃ SINH VIÊN
        print("[7/8] Đọc mã đề thi và mã sinh viên...")
        try:
            # Cắt vùng ROI sử dụng tọa độ từ config
            vung_ma_de = extract_exam_code_region(anh_nan_chinh)
            vung_mssv = extract_student_id_region(anh_nan_chinh)
            
            # Lưu ảnh vùng ROI để debug
            if save_images:
                anh_roi = visualize_all_regions(anh_nan_chinh)
                luu_anh_trung_gian(anh_roi, "06_roi_regions.jpg")
                luu_anh_trung_gian(vung_ma_de, "06a_ma_de_region.jpg")
                luu_anh_trung_gian(vung_mssv, "06b_mssv_region.jpg")
            
            # Đọc mã đề thi - thử nhiều phương pháp
            try:
                ma_de = read_exam_code(vung_ma_de, num_digits=3, threshold_method="adaptive")
            except ValueError as e1:
                print(f"      ⚠️  Adaptive failed: {e1}")
                try:
                    ma_de = read_exam_code(vung_ma_de, num_digits=3, threshold_method="otsu")
                except ValueError as e2:
                    print(f"      ⚠️  Otsu failed: {e2}")
                    try:
                        ma_de = read_exam_code(vung_ma_de, num_digits=3, threshold_method="binary")
                    except ValueError as e3:
                        print(f"      ⚠️  Binary failed: {e3}")
                        ma_de = "N/A"
            
            # Đọc mã sinh viên (6 chữ số)
            try:
                ma_sinh_vien = read_student_id(vung_mssv, num_digits=6, threshold_method="adaptive")
            except ValueError:
                try:
                    ma_sinh_vien = read_student_id(vung_mssv, num_digits=6, threshold_method="otsu")
                except ValueError:
                    try:
                        ma_sinh_vien = read_student_id(vung_mssv, num_digits=6, threshold_method="binary")
                    except ValueError:
                        ma_sinh_vien = "N/A"
            
            print(f"      → Mã đề: {ma_de}")
            print(f"      → Mã sinh viên: {ma_sinh_vien}")
            print(f"      → Xem ảnh ROI tại: output/06a_ma_de_region.jpg và output/06b_mssv_region.jpg")
            
        except Exception as e:
            print(f"      ⚠️  Lỗi: {e}")
            ma_de = "N/A"
            ma_sinh_vien = "N/A"
        
        # Bước 8 - CHẤM ĐIỂM (TODO: Chờ module grader hoàn thành)
        print("[8/8] Chấm điểm...", end=" ")
        print("⚠️  (Module grader chưa hoàn thành)")
        
        print()
        print("=" * 60)
        print("HOÀN THÀNH TIỀN XỬ LÝ VÀ NẮN CHỈNH")
        print("=" * 60)
        
        if save_images:
            print(f"✓ Các ảnh trung gian đã được lưu vào thư mục: output/")
        
        print()
        print("⚠️  LƯU Ý: Module chấm điểm (grader.py) chưa được triển khai.")
        print("   Sau khi hoàn thành grader.py, hệ thống sẽ tự động chấm điểm.")
        print()
        
    except FileNotFoundError as e:
        print(f"\n❌ LỖI: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ LỖI: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ LỖI KHÔNG XÁC ĐỊNH: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def chon_file_anh() -> Optional[str]:
    """
    Mở hộp thoại để chọn file ảnh.
    
    Returns:
        Đường dẫn đến file ảnh được chọn, hoặc None nếu hủy
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh bài thi trắc nghiệm",
        initialdir="data/raw",
        filetypes=[
            ("Tất cả ảnh", "*.jpg *.jpeg *.png *.bmp"),
            ("JPEG", "*.jpg *.jpeg"),
            ("PNG", "*.png"),
            ("BMP", "*.bmp"),
            ("Tất cả files", "*.*")
        ]
    )
    
    root.destroy()
    return file_path if file_path else None


def chon_file_dap_an() -> Optional[str]:
    """
    Mở hộp thoại để chọn file đáp án JSON.
    
    Returns:
        Đường dẫn đến file đáp án, hoặc None nếu hủy
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    file_path = filedialog.askopenfilename(
        title="Chọn file đáp án (tùy chọn)",
        initialdir="data/answer_keys",
        filetypes=[
            ("JSON files", "*.json"),
            ("Tất cả files", "*.*")
        ]
    )
    
    root.destroy()
    return file_path if file_path else None


if __name__ == "__main__":
    # Chế độ 1: Không có tham số -> Mở GUI để chọn file
    if len(sys.argv) == 1:
        print("=" * 60)
        print("HỆ THỐNG CHẤM TRẮC NGHIỆM TỰ ĐỘNG (OMR)")
        print("=" * 60)
        print()
        print("Vui lòng chọn ảnh bài thi...")
        
        image_path = chon_file_anh()
        
        if not image_path:
            print("Không có file nào được chọn. Thoát chương trình.")
            sys.exit(0)
        
        print(f"✓ Đã chọn: {image_path}")
        print()
        
        # Hỏi có muốn chọn file đáp án không
        print("Bạn có muốn chọn file đáp án tùy chỉnh không?")
        print("   (Nhấn Enter để dùng đáp án mẫu, hoặc nhập 'y' để chọn file)")
        chon = input("   Lựa chọn: ").strip().lower()
        
        answer_key_path = None
        if chon in ['y', 'yes', 'có', 'c']:
            print()
            print("Vui lòng chọn file đáp án JSON...")
            answer_key_path = chon_file_dap_an()
            if answer_key_path:
                print(f"✓ Đã chọn: {answer_key_path}")
            else:
                print("Không chọn file đáp án. Sẽ dùng đáp án mẫu.")
        else:
            print("Sẽ dùng đáp án mẫu từ config.py")
        
        print()
        main(image_path, answer_key_path, save_images=True)
    
    # Chế độ 2: Có tham số -> Dùng command-line
    else:
        image_path = sys.argv[1]
        answer_key_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        main(image_path, answer_key_path, save_images=True)
