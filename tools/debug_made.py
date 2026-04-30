"""Debug mã đề reading cho phiếu có bubble số (01/02/03)."""
import cv2
import numpy as np
from src.reader import phat_hien_anchor, phan_loai_vung_roi, _cat_vung_roi, _phan_nguong

img = cv2.imread('E:/cham-trac-nghiem-xla/data/raw/test_sheet_01.jpg')
from src.transform import tim_canh, tim_goc_giay, nan_chinh_anh
from src.preprocessing import chuyen_xam, loc_nhieu

anh_xam = chuyen_xam(img)
anh_mo = loc_nhieu(anh_xam, loai_loc="gaussian", kich_thuoc=5)
anh_canh = tim_canh(anh_mo)
cac_goc = tim_goc_giay(anh_canh, auto_detect_cropped=True)
if cac_goc is not None:
    img_w = nan_chinh_anh(img, cac_goc, 800, 1200)
else:
    img_w = cv2.resize(img, (800, 1200))

anchors = phat_hien_anchor(img_w)
rois = phan_loai_vung_roi(anchors, 1200, 800)
roi_md = rois['ma_de']
print('ROI ma de:', roi_md)

# Cắt vùng mã đề
md_crop = _cat_vung_roi(img_w, *roi_md)
cv2.imwrite('output/debug_made_full.jpg', md_crop)

# Thử cắt chỉ phần bubble (bỏ text dọc bên phải)
# Phần text "FILLING..." thường chiếm ~20% chiều rộng bên phải
rx, ry, rw, rh = roi_md
md_no_text_w = int(rw * 0.7)  # chỉ lấy 70% bên trái
md_no_text = img_w[ry:ry+rh, rx:rx+rx+md_no_text_w]
cv2.imwrite('output/debug_made_no_text.jpg', img_w[ry:ry+rh, rx:rx+md_no_text_w])

# Đọc pixel counts với adaptive threshold
gray = cv2.cvtColor(md_crop, cv2.COLOR_BGR2GRAY) if md_crop.ndim == 3 else md_crop
binary_adapt = _phan_nguong(gray, 'adaptive')
binary_otsu = _phan_nguong(gray, 'otsu')

# Tìm circles trong vùng mã đề
blur = cv2.GaussianBlur(gray, (5,5), 0)
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=12, 
    param1=50, param2=20, minRadius=6, maxRadius=16)
if circles is not None:
    circles = np.round(circles[0]).astype(int)
    print(f'Circles in ma de: {len(circles)}')
    from src.reader import _cluster_1d_simple
    all_cx = sorted([int(c[0]) for c in circles])
    all_cy = sorted([int(c[1]) for c in circles])
    cols = _cluster_1d_simple(all_cx, min_gap=10)
    rows = _cluster_1d_simple(all_cy, min_gap=10)
    print('cols x:', cols)
    print('rows y:', rows)
    
    # Pixel counts từng bubble (OTSU)
    avg_r = int(np.mean([c[2] for c in circles]))
    bw = bh = avg_r * 2 + 4
    h, w = binary_otsu.shape[:2]
    for col_i, cx in enumerate(cols[:3]):  # chỉ 3 cột đầu
        counts = []
        for cy in rows[-10:]:  # 10 hàng cuối
            x1 = max(0, cx - bw//2); x2 = min(w, x1+bw)
            y1 = max(0, cy - bh//2); y2 = min(h, y1+bh)
            counts.append(cv2.countNonZero(binary_otsu[y1:y2, x1:x2]))
        print(f'Col {col_i} (x={cx}): {counts}')
else:
    print('No circles found in ma de region')
