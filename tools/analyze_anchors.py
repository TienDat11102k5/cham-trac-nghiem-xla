"""
Script phân tích vị trí các ô vuông đen (anchor markers) trên phiếu trắc nghiệm.
Mục đích: Xác định chính xác layout phiếu để xây dựng thuật toán auto-detect ROI.
"""
import cv2
import numpy as np
import sys
import os

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import doc_anh, chuyen_xam, loc_nhieu
from src.transform import tim_canh, tim_goc_giay, nan_chinh_anh


def phan_tich_anchor(image_path: str) -> None:
    """Phân tích và vẽ các anchor markers trên ảnh nắn chỉnh."""
    
    # Bước 1-6: Pipeline tiền xử lý (giữ nguyên logic main.py)
    print(f"Đang xử lý: {image_path}")
    anh_goc = doc_anh(image_path)
    anh_xam = chuyen_xam(anh_goc)
    anh_mo = loc_nhieu(anh_xam, loai_loc="gaussian", kich_thuoc=5)
    anh_canh = tim_canh(anh_mo, nguong_thap=50, nguong_cao=150)
    cac_goc = tim_goc_giay(anh_canh, auto_detect_cropped=True)
    
    if cac_goc is None:
        anh_nan_chinh = cv2.resize(anh_goc, (800, 1200))
    else:
        anh_nan_chinh = nan_chinh_anh(anh_goc, cac_goc, chieu_rong=800, chieu_cao=1200)
    
    print(f"Ảnh nắn chỉnh: {anh_nan_chinh.shape}")
    
    # Bước 7: Phân tích anchor
    gray = chuyen_xam(anh_nan_chinh)
    
    # Thử nhiều phương pháp threshold
    for method_name, binary in [
        ("Otsu", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]),
        ("Adaptive", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)),
    ]:
        print(f"\n{'='*60}")
        print(f"Phương pháp: {method_name}")
        print(f"{'='*60}")
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Tìm contours
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Lọc contour hình vuông đen
        candidates = []
        img_h, img_w = gray.shape[:2]
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Lọc theo diện tích (anchor khoảng 15x15 đến 40x40 pixel trên ảnh 800x1200)
            if area < 100 or area > 2500:
                continue
            
            # Lọc theo tỉ lệ cạnh (gần vuông)
            aspect = w / h if h > 0 else 0
            if aspect < 0.5 or aspect > 2.0:
                continue
            
            # Lọc theo solidity (đặc, không rỗng)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity < 0.7:
                continue
            
            # Lọc theo extent (tỉ lệ diện tích contour / bounding rect)
            extent = area / (w * h)
            if extent < 0.6:
                continue
            
            candidates.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'cx': x + w // 2, 'cy': y + h // 2,
                'area': area, 'aspect': aspect,
                'solidity': solidity, 'extent': extent
            })
        
        print(f"Tìm thấy {len(candidates)} ứng viên anchor")
        
        # Sắp xếp theo vị trí y
        candidates.sort(key=lambda c: c['cy'])
        
        for i, c in enumerate(candidates):
            print(f"  #{i+1}: ({c['x']}, {c['y']}) size={c['w']}x{c['h']} "
                  f"area={c['area']} aspect={c['aspect']:.2f} "
                  f"solidity={c['solidity']:.2f} extent={c['extent']:.2f}")
        
        # Vẽ lên ảnh
        vis = anh_nan_chinh.copy()
        for i, c in enumerate(candidates):
            cv2.rectangle(vis, (c['x'], c['y']), (c['x']+c['w'], c['y']+c['h']), (0, 0, 255), 2)
            cv2.putText(vis, f"#{i+1}", (c['x'], c['y']-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        out_name = f"output/debug_anchors_{method_name.lower()}.jpg"
        cv2.imwrite(out_name, vis)
        print(f"Đã lưu: {out_name}")


if __name__ == "__main__":
    for img in ["data/raw/test_sheet_01.jpg", "data/raw/test_sheet_02.jpg"]:
        print(f"\n{'#'*60}")
        print(f"# ẢNH: {img}")
        print(f"{'#'*60}")
        phan_tich_anchor(img)
