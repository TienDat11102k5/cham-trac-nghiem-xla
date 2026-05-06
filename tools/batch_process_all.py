"""
Script batch processing - Chạy main.py cho tất cả ảnh test_sheet
Tự động xử lý 15 ảnh (test_sheet_01 đến test_sheet_17, thiếu 02)
"""

import sys
import os
from pathlib import Path

# Thêm thư mục gốc vào sys.path để import được các module
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from main import main


def batch_process_all_sheets():
    """
    Chạy pipeline OMR cho tất cả ảnh test_sheet trong data/raw/
    """
    
    # Đường dẫn
    raw_folder = project_root / "data" / "raw"
    answer_key_path = project_root / "data" / "answer_keys" / "all_answer_keys.json"
    
    # Kiểm tra file đáp án
    if not answer_key_path.exists():
        print(f"KHÔNG TÌM THẤY FILE ĐÁP ÁN: {answer_key_path}")
        return
    
    # Danh sách ảnh cần xử lý (01-17, thiếu 02)
    sheet_numbers = [
        "01", "03", "04", "05", "06", "07", "08", "09", "10",
        "11", "12", "13", "14", "15", "16", "17"
    ]
    
    print("=" * 80)
    print("BATCH PROCESSING - CHẠY TẤT CẢ ẢNH TEST_SHEET")
    print("=" * 80)
    print(f"Thư mục ảnh: {raw_folder}")
    print(f"File đáp án: {answer_key_path}")
    print(f"Số lượng ảnh: {len(sheet_numbers)}")
    print("=" * 80)
    print()
    
    # Thống kê
    success_count = 0
    failed_count = 0
    failed_sheets = []
    
    # Xử lý từng ảnh
    for i, sheet_num in enumerate(sheet_numbers, 1):
        image_name = f"test_sheet_{sheet_num}.jpg"
        image_path = raw_folder / image_name
        
        print(f"\n{'=' * 80}")
        print(f"[{i}/{len(sheet_numbers)}] XỬ LÝ: {image_name}")
        print(f"{'=' * 80}")
        
        # Kiểm tra file tồn tại
        if not image_path.exists():
            print(f"❌ KHÔNG TÌM THẤY FILE: {image_path}")
            failed_count += 1
            failed_sheets.append(image_name)
            continue
        
        # Chạy pipeline
        try:
            main(
                image_path=str(image_path),
                answer_key_path=str(answer_key_path),
                save_images=True
            )
            success_count += 1
            print(f"✅ THÀNH CÔNG: {image_name}")
            
        except Exception as e:
            print(f"❌ LỖI: {image_name}")
            print(f"   Chi tiết: {e}")
            failed_count += 1
            failed_sheets.append(image_name)
    
    # Tổng kết
    print("\n" + "=" * 80)
    print("TỔNG KẾT BATCH PROCESSING")
    print("=" * 80)
    print(f"✅ Thành công: {success_count}/{len(sheet_numbers)}")
    print(f"❌ Thất bại: {failed_count}/{len(sheet_numbers)}")
    
    if failed_sheets:
        print(f"\nDanh sách ảnh thất bại:")
        for sheet in failed_sheets:
            print(f"  - {sheet}")
    
    print("\n" + "=" * 80)
    print("KẾT QUẢ:")
    print(f"  - Ảnh trung gian: output/")
    print(f"  - File JSON: data/answer_keys/result_test_sheet_XX.json")
    print("=" * 80)


if __name__ == "__main__":
    batch_process_all_sheets()
