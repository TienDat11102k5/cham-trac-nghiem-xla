import cv2
import numpy as np
from src.reader import phat_hien_anchor, phan_loai_vung_roi

# Chạy lại sheet 01 để lưu ảnh trung gian mới nhất
img = cv2.imread('E:/cham-trac-nghiem-xla/data/raw/test_sheet_01.jpg')
print('img shape:', img.shape)

# Resize giống main.py
h_goc, w_goc = img.shape[:2]
img_resized = cv2.resize(img, (800, 1200))
print('resized shape:', img_resized.shape)

anchors = phat_hien_anchor(img_resized)
print(f'anchors: {len(anchors)}')
for i, a in enumerate(anchors):
    print(f'  #{i+1}: cx={a["cx"]:.0f} cy={a["cy"]:.0f} w={a["w"]} h={a["h"]} area={a["area"]:.0f}')

rois = phan_loai_vung_roi(anchors, 1200, 800)
print('SBD:', rois['sbd'])
print('Ma de:', rois['ma_de'])
print('Dap an:', rois['dap_an'])

# Lưu vùng cắt để kiểm tra
rx, ry, rw, rh = rois['ma_de']
cv2.imwrite('output/debug_01_made.jpg', img_resized[ry:ry+rh, rx:rx+rw])
rx2, ry2, rw2, rh2 = rois['dap_an']
cv2.imwrite('output/debug_01_dapan.jpg', img_resized[ry2:ry2+rh2, rx2:rx2+rw2])
rx3, ry3, rw3, rh3 = rois['sbd']
cv2.imwrite('output/debug_01_sbd.jpg', img_resized[ry3:ry3+rh3, rx3:rx3+rw3])

# Visualize anchors
from src.reader import visualize_anchors, visualize_all_regions
anh_anchor = visualize_anchors(img_resized, anchors)
cv2.imwrite('output/debug_01_anchors.jpg', anh_anchor)
anh_roi = visualize_all_regions(img_resized, rois=rois)
cv2.imwrite('output/debug_01_roi.jpg', anh_roi)
print('Done')
