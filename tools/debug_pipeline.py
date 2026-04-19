"""
Tool debug tổng hợp cho pipeline OMR.

Sử dụng:
    py tools/debug_pipeline.py <image_path>

Ví dụ:
    py tools/debug_pipeline.py data/raw/test_sheet_01.jpg

Output:
    output/debug_roi_regions.jpg   — Vùng ROI (SBD, Mã đề, Đáp án)
    output/debug_grid_answers.jpg  — Grid bubble + đáp án đã chọn
    output/debug_sbd_grid.jpg      — Grid SBD + bubble tô
    output/debug_made_grid.jpg     — Grid Mã đề + bubble tô
"""
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import doc_anh, chuyen_xam, loc_nhieu
from src.transform import tim_canh, tim_goc_giay, nan_chinh_anh
from src.reader import (
    phat_hien_anchor, phan_loai_vung_roi, visualize_all_regions,
    read_student_id, read_exam_code
)
from src.grader import segment_bubbles, phat_hien_luoi_bubble


CHOICES = ["A", "B", "C", "D"]


def debug_pipeline(image_path: str) -> None:
    """Chạy pipeline debug toàn bộ."""
    os.makedirs("output", exist_ok=True)

    # ── Pipeline tiền xử lý ──
    print(f"[1] Đọc ảnh: {image_path}")
    anh = doc_anh(image_path)
    xam = chuyen_xam(anh)
    mo = loc_nhieu(xam, "gaussian", 5)
    canh = tim_canh(mo, 50, 150)

    print("[2] Tìm góc + nắn chỉnh...")
    goc = tim_goc_giay(canh, True)
    warp = nan_chinh_anh(anh, goc, 800, 1200)
    gray = chuyen_xam(warp)
    binary = segment_bubbles(gray, "adaptive")

    # ── Anchor + ROI ──
    print("[3] Phát hiện anchor...")
    anchors = phat_hien_anchor(warp)
    rois = phan_loai_vung_roi(anchors, *warp.shape[:2])
    print(f"    Anchors: {len(anchors)}")
    for key, val in rois.items():
        print(f"    {key}: {val}")

    vis_roi = visualize_all_regions(warp, rois)
    cv2.imwrite("output/debug_roi_regions.jpg", vis_roi)

    # ── SBD ──
    print("[4] Đọc SBD...")
    sx, sy, sw, sh = rois["sbd"]
    sbd_roi = warp[sy:sy + sh, sx:sx + sw]
    try:
        sbd = read_student_id(sbd_roi)
        print(f"    SBD = {sbd}")
    except ValueError as e:
        sbd = "N/A"
        print(f"    SBD lỗi: {e}")

    # Debug grid SBD
    _debug_code_grid(gray[sy:sy + sh, sx:sx + sw], "output/debug_sbd_grid.jpg", num_digits=6)

    # ── Mã đề ──
    print("[5] Đọc mã đề...")
    mx, my, mw, mh = rois["ma_de"]
    md_roi = warp[my:my + mh, mx:mx + mw]
    try:
        ma_de = read_exam_code(md_roi)
        print(f"    Mã đề = {ma_de}")
    except ValueError as e:
        ma_de = "N/A"
        print(f"    Mã đề lỗi: {e}")

    _debug_code_grid(gray[my:my + mh, mx:mx + mw], "output/debug_made_grid.jpg", num_digits=3)

    # ── Đáp án ──
    print("[6] Đọc đáp án...")
    roi = rois["dap_an"]
    cl, cr, ry = phat_hien_luoi_bubble(binary, roi, 20, 4, gray_image=gray)

    bh = max(1, int(abs(ry[1] - ry[0]) * 0.7)) if len(ry) >= 2 else 27
    bw = max(1, int(abs(cl[1] - cl[0]) * 0.8)) if len(cl) >= 2 else 31

    vis = warp.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    print(f"\n{'Câu':>4} | {'A':>5} {'B':>5} {'C':>5} {'D':>5} | Chọn")
    print("-" * 45)

    for i in range(len(ry)):
        y = ry[i]

        for cols, q_offset in [(cl, 0), (cr, 10)]:
            q = i + 1 + q_offset
            counts = []
            for x in cols:
                region = binary[max(0, y):min(1200, y + bh),
                                max(0, x):min(800, x + bw)]
                counts.append(cv2.countNonZero(region))

            max_idx = int(np.argmax(counts))
            ans = CHOICES[max_idx]
            print(f"{q:4d} | {counts[0]:5d} {counts[1]:5d} {counts[2]:5d} {counts[3]:5d} | {ans}")

            # Vẽ lên ảnh
            for j, x in enumerate(cols):
                color = (0, 0, 255) if j == max_idx else (0, 255, 0)
                thick = 2 if j == max_idx else 1
                cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, thick)

    cv2.imwrite("output/debug_grid_answers.jpg", vis)
    print(f"\n✓ Saved debug images to output/")


def _debug_code_grid(gray_roi: np.ndarray, save_path: str, num_digits: int) -> None:
    """Vẽ HoughCircles grid lên ảnh ROI."""
    from src.reader import _cluster_1d_simple

    blur = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=15,
        param1=50, param2=25,
        minRadius=8, maxRadius=16
    )

    vis = cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2BGR)

    if circles is not None:
        circles_arr = np.round(circles[0]).astype(int)
        all_cx = sorted([int(c[0]) for c in circles_arr])
        col_centers = _cluster_1d_simple(all_cx, min_gap=15)
        all_cy = sorted([int(c[1]) for c in circles_arr])
        row_centers = _cluster_1d_simple(all_cy, min_gap=15)

        if len(row_centers) > 10:
            row_centers = row_centers[-10:]

        avg_r = int(np.mean([int(c[2]) for c in circles_arr]))
        bw = avg_r * 2 + 4
        bh = avg_r * 2 + 4

        for ry in row_centers:
            for cx in col_centers[:num_digits]:
                x1 = int(cx - bw // 2)
                y1 = int(ry - bh // 2)
                cv2.rectangle(vis, (x1, y1), (x1 + bw, y1 + bh), (0, 255, 0), 1)

    cv2.imwrite(save_path, vis)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Sử dụng: py tools/debug_pipeline.py <image_path>")
        print("Ví dụ:   py tools/debug_pipeline.py data/raw/test_sheet_01.jpg")
        sys.exit(1)

    debug_pipeline(sys.argv[1])
