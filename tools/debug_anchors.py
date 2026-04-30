"""
Debug: phân tích TẤT CẢ contour để tìm ngưỡng phù hợp cho 11 anchors.
Chạy: python tools/debug_anchors.py data/raw/test_sheet_02.jpg
"""
import sys
import cv2
import numpy as np
from pathlib import Path

def debug_anchors(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Không đọc được ảnh: {image_path}")
        return

    h_img, w_img = img.shape[:2]
    print(f"Kích thước ảnh: {w_img}x{h_img}")

    # --- Tiền xử lý giống pipeline chính ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Tổng số contour: {len(contours)}")

    # --- Phân tích từng contour trong dải diện tích rộng hơn ---
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100 or area > 2000:   # Mở rộng dải tìm kiếm
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 999

        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx = x + w / 2.0
            cy = y + h / 2.0

        candidates.append({
            'x': x, 'y': y, 'w': w, 'h': h,
            'cx': cx, 'cy': cy,
            'area': area, 'extent': extent, 'solidity': solidity,
            'aspect': aspect
        })

    # Sắp xếp theo area descending
    candidates.sort(key=lambda c: c['area'], reverse=True)

    print(f"\nTổng contour trong dải [100, 2000] px²: {len(candidates)}")
    print(f"\n{'No':>3} {'cx':>6} {'cy':>6} {'w':>4} {'h':>4} {'area':>6} "
          f"{'extent':>7} {'solid':>7} {'aspect':>7}  Ghi chú")
    print("-" * 80)

    # Ngưỡng hiện tại
    CURR_MIN_AREA    = 250
    CURR_MAX_AREA    = 600
    CURR_MIN_EXTENT  = 0.82
    CURR_MIN_SOLID   = 0.93

    passed_current = []
    for i, c in enumerate(candidates):
        ok_area    = CURR_MIN_AREA <= c['area'] <= CURR_MAX_AREA
        ok_extent  = c['extent']   >= CURR_MIN_EXTENT
        ok_solid   = c['solidity'] >= CURR_MIN_SOLID
        passes     = ok_area and ok_extent and ok_solid
        if passes:
            passed_current.append(c)

        flag = "✓ PASS" if passes else (
            "  area?" if not ok_area else
            "  ext? " if not ok_extent else
            "  sol? "
        )
        print(f"{i+1:>3} {c['cx']:>6.1f} {c['cy']:>6.1f} {c['w']:>4} {c['h']:>4} "
              f"{c['area']:>6.0f} {c['extent']:>7.3f} {c['solidity']:>7.3f} "
              f"{c['aspect']:>7.2f}  {flag}")

    print(f"\n{'='*80}")
    print(f"Ngưỡng hiện tại: area=[{CURR_MIN_AREA},{CURR_MAX_AREA}], "
          f"extent>={CURR_MIN_EXTENT}, solidity>={CURR_MIN_SOLID}")
    print(f"Số anchor PASS: {len(passed_current)} / {len(candidates)}")

    # --- Vẽ debug ảnh ---
    vis = img.copy()

    # Tất cả candidate (màu vàng mờ)
    for c in candidates:
        cv2.rectangle(vis, (c['x'], c['y']),
                      (c['x']+c['w'], c['y']+c['h']), (0, 200, 200), 1)

    # Chỉ những cái PASS (màu đỏ đậm)
    for j, c in enumerate(passed_current):
        cv2.rectangle(vis, (c['x'], c['y']),
                      (c['x']+c['w'], c['y']+c['h']), (0, 0, 255), 2)
        cv2.putText(vis, f"#{j+1}", (c['x'], c['y']-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    out_path = Path("output") / "debug_all_candidates.jpg"
    cv2.imwrite(str(out_path), vis)
    print(f"\nĐã lưu ảnh debug: {out_path}")

    # --- Gợi ý ngưỡng mới ---
    print("\n--- GỢI Ý NGƯỠNG MỚI ---")
    # Tìm những contour có extent >= 0.65 và solidity >= 0.80 (nới lỏng hơn)
    relaxed = [c for c in candidates
               if c['extent'] >= 0.65 and c['solidity'] >= 0.80 and c['aspect'] <= 2.0]
    print(f"Với extent>=0.65, solidity>=0.80, aspect<=2.0: {len(relaxed)} contour")
    areas = [c['area'] for c in relaxed]
    if areas:
        print(f"  area range: [{min(areas):.0f}, {max(areas):.0f}]")
        print(f"  extents: {sorted(set(round(c['extent'],3) for c in relaxed))}")
        print(f"  solidities: {sorted(set(round(c['solidity'],3) for c in relaxed))}")

    # Vẽ "relaxed" pass (màu xanh)
    vis2 = img.copy()
    for j, c in enumerate(relaxed):
        cv2.rectangle(vis2, (c['x'], c['y']),
                      (c['x']+c['w'], c['y']+c['h']), (255, 0, 0), 2)
        cv2.putText(vis2, f"R{j+1}", (c['x'], c['y']-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    out_path2 = Path("output") / "debug_relaxed_candidates.jpg"
    cv2.imwrite(str(out_path2), vis2)
    print(f"Đã lưu ảnh relaxed: {out_path2}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Mặc định dùng file resize đã có trong output
        import os
        candidates_imgs = [
            "output/05_resize_800x1200.jpg",
            "data/raw/test_sheet_02.jpg",
            "data/raw/test_sheet_01.jpg",
        ]
        path = next((p for p in candidates_imgs if os.path.exists(p)), None)
        if path is None:
            print("Không tìm thấy ảnh test. Truyền đường dẫn ảnh vào argument.")
            sys.exit(1)
        print(f"Dùng ảnh: {path}")
        debug_anchors(path)
    else:
        debug_anchors(sys.argv[1])
