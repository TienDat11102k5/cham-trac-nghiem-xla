import sys
import io

# Fix Unicode encoding trên Windows terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import tkinter as tk
from tkinter import filedialog

from src.preprocessing import doc_anh, chuyen_xam, loc_nhieu
from src.transform import tim_canh, tim_goc_giay, nan_chinh_anh
from src.reader import (
    phat_hien_anchor, phan_loai_vung_roi,
    extract_exam_code_region, read_exam_code,
    extract_student_id_region, read_student_id,
    extract_answer_region,
    visualize_all_regions, visualize_anchors,
)
from src.grader import grade_from_image


def doc_dap_an(answer_key_path: str, exam_id: str = None) -> tuple:
    with open(answer_key_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Nếu file chứa nhiều mã đề (all_answer_keys.json)
    if exam_id and exam_id in data:
        exam_data = data[exam_id]
        answers = exam_data.get("answers", {})
        exam_id_found = exam_data.get("exam_id", exam_id)
        answer_dict = {int(k): v for k, v in answers.items()}
        return answer_dict, exam_id_found
    
    # Nếu file chỉ chứa 1 mã đề
    if "answers" in data:
        answers = data["answers"]
        exam_id_found = data.get("exam_id", "N/A")
    else:
        answers = data
        exam_id_found = "N/A"
    
    answer_dict = {int(k): v for k, v in answers.items()}
    return answer_dict, exam_id_found


def luu_anh_trung_gian(anh: any, ten_file: str, thu_muc: str = "output") -> None:
    Path(thu_muc).mkdir(parents=True, exist_ok=True)
    duong_dan = Path(thu_muc) / ten_file
    cv2.imwrite(str(duong_dan), anh)


def xoa_anh_output(thu_muc: str = "output") -> None:
    output_path = Path(thu_muc)
    if output_path.exists():
        for file in output_path.glob("*.jpg"):
            file.unlink()
        for file in output_path.glob("*.png"):
            file.unlink()


def main(image_path: str, answer_key_path: Optional[str] = None, save_images: bool = True) -> None:
    xoa_anh_output()
    
    print("=" * 60)
    print("HE THONG CHAM TRAC NGHIEM TU DONG (OMR)")
    print("=" * 60)
    print(f"Dang xu ly: {image_path}")
    print()
    
    try:
        print("[1/8] Doc anh...", end=" ")
        anh_goc = doc_anh(image_path)
        print("OK")
        
        print("[2/8] Chuyen sang anh xam...", end=" ")
        anh_xam = chuyen_xam(anh_goc)
        if save_images:
            luu_anh_trung_gian(anh_xam, "01_anh_xam.jpg")
        print("OK")
        
        print("[3/8] Ap dung bo loc khu nhieu (Gaussian)...", end=" ")
        anh_mo = loc_nhieu(anh_xam, loai_loc="gaussian", kich_thuoc=5)
        if save_images:
            luu_anh_trung_gian(anh_mo, "02_loc_nhieu.jpg")
        print("OK")
        
        print("[4/8] Phat hien bien (Canny)...", end=" ")
        anh_canh = tim_canh(anh_mo, nguong_thap=50, nguong_cao=150)
        if save_images:
            luu_anh_trung_gian(anh_canh, "03_canh.jpg")
        print("OK")
        
        print("[5/8] Tim 4 goc toa giay thi...", end=" ")
        cac_goc = tim_goc_giay(anh_canh, auto_detect_cropped=True)
        
        if cac_goc is None:
            print("(Anh da cat san)")
            print("      Resize anh ve 800x1200...", end=" ")
            h_goc, w_goc = anh_goc.shape[:2]
            anh_nan_chinh = cv2.resize(anh_goc, (800, 1200))
            print(f"OK (tu {w_goc}x{h_goc})")
            if save_images:
                luu_anh_trung_gian(anh_nan_chinh, "05_resize_800x1200.jpg")
        else:
            print("OK")
            
            if save_images:
                anh_ve_goc = anh_goc.copy()
                for i, goc in enumerate(cac_goc):
                    cv2.circle(anh_ve_goc, tuple(goc.astype(int)), 10, (0, 255, 0), -1)
                    cv2.putText(anh_ve_goc, str(i), tuple(goc.astype(int)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                luu_anh_trung_gian(anh_ve_goc, "04_tim_goc.jpg")
            
            print("[6/8] Nan chinh anh (Perspective Transform)...", end=" ")
            anh_nan_chinh = nan_chinh_anh(anh_goc, cac_goc, chieu_rong=800, chieu_cao=1200)
            if save_images:
                luu_anh_trung_gian(anh_nan_chinh, "05_nan_chinh.jpg")
            print("OK")
        
        print("[7/8] Tu dong phat hien vung ROI...")
        try:
            anchors = phat_hien_anchor(anh_nan_chinh)
            print(f"      Tim thay {len(anchors)} anchor markers")
            
            rois = phan_loai_vung_roi(anchors, *anh_nan_chinh.shape[:2])
            print(f"      SBD: {rois['sbd']}")
            print(f"      Ma de: {rois['ma_de']}")
            print(f"      Dap an: {rois['dap_an']}")
            
            vung_ma_de = extract_exam_code_region(anh_nan_chinh,
                *rois['ma_de'])
            vung_mssv = extract_student_id_region(anh_nan_chinh,
                *rois['sbd'])
            vung_dap_an = extract_answer_region(anh_nan_chinh,
                *rois['dap_an'])
            
            if save_images:
                anh_roi = visualize_all_regions(anh_nan_chinh, rois=rois)
                luu_anh_trung_gian(anh_roi, "06_roi_regions.jpg")
                anh_anchor = visualize_anchors(anh_nan_chinh, anchors)
                luu_anh_trung_gian(anh_anchor, "06a_anchors.jpg")
                luu_anh_trung_gian(vung_ma_de, "06b_ma_de_region.jpg")
                luu_anh_trung_gian(vung_mssv, "06c_mssv_region.jpg")
                luu_anh_trung_gian(vung_dap_an, "06d_dap_an_region.jpg")
            
            ma_de = "N/A"
            for method in ["otsu", "adaptive", "binary"]:
                try:
                    ma_de = read_exam_code(vung_ma_de, num_digits=3,
                                           threshold_method=method)
                    break
                except ValueError as e:
                    print(f"      Doc ma de ({method}): {e}")
            
            ma_sinh_vien = "N/A"
            for method in ["otsu", "adaptive", "binary"]:
                try:
                    ma_sinh_vien = read_student_id(vung_mssv, num_digits=6,
                                                   threshold_method=method)
                    break
                except ValueError:
                    pass
            
            print(f"      Ma de: {ma_de}")
            print(f"      So bao danh: {ma_sinh_vien}")
            
        except Exception as e:
            print(f"      Loi phat hien ROI: {e}")
            ma_de = "N/A"
            ma_sinh_vien = "N/A"
            rois = None
        
        print("[8/8] Cham diem...")
        
        if answer_key_path:
            dap_an, exam_id_expected = doc_dap_an(answer_key_path, exam_id=ma_de)
        else:
            print("      Khong co file dap an. Vui long chon file dap an JSON.")
            answer_key_path = chon_file_dap_an()
            if answer_key_path:
                dap_an, exam_id_expected = doc_dap_an(answer_key_path, exam_id=ma_de)
            else:
                print("      Khong co file dap an. Khong the cham diem.")
                return
        
        print(f"      Ma de tren file dap an: {exam_id_expected}")
        print(f"      Ma de doc duoc: {ma_de}")
        
        if ma_de != "N/A" and exam_id_expected != "N/A":
            if ma_de == exam_id_expected:
                print("      OK - Ma de khop!")
            else:
                print(f"      CANH BAO - Ma de khong khop! Hy vong: {exam_id_expected}, Doc duoc: {ma_de}")
        
        ten_anh = Path(image_path).stem
        json_output = f"data/answer_keys/result_{ten_anh}.json"
        
        try:
            correct_count, score, student_answers = grade_from_image(
                warped_image=anh_nan_chinh,
                answer_key=dap_an,
                num_questions=20,
                save_json=json_output,
                image_path=image_path,
                so_bao_danh=ma_sinh_vien,
                ma_de=ma_de
            )
        except Exception as e:
            print(f"      Loi cham diem: {e}")
            import traceback
            traceback.print_exc()
        
        print()
        print("=" * 60)
        print("HOAN THANH PIPELINE OMR")
        print("=" * 60)
        
        if save_images:
            print(f"Anh trung gian: output/")
        print(f"Ket qua JSON: {json_output}")
        print()
        
    except FileNotFoundError as e:
        print(f"\nLOI: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\nLOI: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nLOI KHONG XAC DINH: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def chon_file_anh() -> Optional[str]:
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    file_path = filedialog.askopenfilename(
        title="Chon anh bai thi trac nghiem",
        initialdir="data/raw",
        filetypes=[
            ("Tat ca anh", "*.jpg *.jpeg *.png *.bmp"),
            ("JPEG", "*.jpg *.jpeg"),
            ("PNG", "*.png"),
            ("BMP", "*.bmp"),
            ("Tat ca files", "*.*")
        ]
    )
    
    root.destroy()
    return file_path if file_path else None


def chon_file_dap_an() -> Optional[str]:
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    file_path = filedialog.askopenfilename(
        title="Chon file dap an (JSON)",
        initialdir="data/answer_keys",
        filetypes=[
            ("JSON files", "*.json"),
            ("Tat ca files", "*.*")
        ]
    )
    
    root.destroy()
    return file_path if file_path else None


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 60)
        print("HE THONG CHAM TRAC NGHIEM TU DONG (OMR)")
        print("=" * 60)
        print()
        print("Vui long chon anh bai thi...")
        
        image_path = chon_file_anh()
        
        if not image_path:
            print("Khong co file nao duoc chon. Thoat chuong trinh.")
            sys.exit(0)
        
        print(f"Da chon: {image_path}")
        print()
        
        print("Vui long chon file dap an JSON...")
        answer_key_path = chon_file_dap_an()
        
        if not answer_key_path:
            print("Khong chon file dap an. Thoat chuong trinh.")
            sys.exit(0)
        
        print(f"Da chon: {answer_key_path}")
        print()
        main(image_path, answer_key_path, save_images=True)
    
    else:
        image_path = sys.argv[1]
        answer_key_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        main(image_path, answer_key_path, save_images=True)
