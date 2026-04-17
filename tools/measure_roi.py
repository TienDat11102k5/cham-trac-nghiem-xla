"""
Script để đo tọa độ ROI trên ảnh đã nắn chỉnh.

Cách dùng:
    python tools/measure_roi.py output/05_nan_chinh.jpg
    
Click chuột để đánh dấu các điểm:
- Click góc trên-trái của vùng cần đo
- Click góc dưới-phải của vùng cần đo
- Nhấn 'q' để thoát
"""

import cv2
import sys
import numpy as np

# Biến global để lưu tọa độ
points = []
image = None
image_display = None
scale_factor = 1.0

def mouse_callback(event, x, y, flags, param):
    global points, image_display, scale_factor
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Chuyển đổi tọa độ từ ảnh hiển thị về ảnh gốc
        orig_x = int(x / scale_factor)
        orig_y = int(y / scale_factor)
        
        points.append((orig_x, orig_y))
        print(f"Điểm {len(points)}: ({orig_x}, {orig_y})")
        
        # Vẽ điểm lên ảnh hiển thị
        cv2.circle(image_display, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(image_display, f"{len(points)}", (x+10, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Nếu có 2 điểm, vẽ hình chữ nhật
        if len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            
            # Vẽ trên ảnh hiển thị (đã scale)
            display_x1 = int(x1 * scale_factor)
            display_y1 = int(y1 * scale_factor)
            display_x2 = int(x2 * scale_factor)
            display_y2 = int(y2 * scale_factor)
            
            cv2.rectangle(image_display, (display_x1, display_y1), (display_x2, display_y2), (0, 255, 0), 2)
            
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            
            print(f"\n{'='*60}")
            print(f"ROI_X = {min(x1, x2)}")
            print(f"ROI_Y = {min(y1, y2)}")
            print(f"ROI_WIDTH = {width}")
            print(f"ROI_HEIGHT = {height}")
            print(f"{'='*60}\n")
            
            # Reset để đo vùng mới
            points.clear()
        
        cv2.imshow("Measure ROI", image_display)

def main():
    global image, image_display, scale_factor
    
    if len(sys.argv) < 2:
        print("Cách dùng: python tools/measure_roi.py <đường_dẫn_ảnh>")
        print("Ví dụ: python tools/measure_roi.py output/05_nan_chinh.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Không đọc được ảnh: {image_path}")
        sys.exit(1)
    
    orig_h, orig_w = image.shape[:2]
    
    # Tính scale để ảnh vừa màn hình (max 800 pixels chiều cao để chắc chắn)
    max_height = 800
    if orig_h > max_height:
        scale_factor = max_height / orig_h
        new_w = int(orig_w * scale_factor)
        new_h = int(orig_h * scale_factor)
        image_display = cv2.resize(image, (new_w, new_h))
        print(f"Đã resize ảnh từ {orig_w}x{orig_h} xuống {new_w}x{new_h} (scale: {scale_factor:.2f})")
    else:
        scale_factor = 1.0
        image_display = image.copy()
    
    print(f"Đã mở ảnh: {image_path}")
    print(f"Kích thước gốc: {orig_w}x{orig_h}")
    print("\nHướng dẫn:")
    print("1. Click góc trên-trái của vùng cần đo")
    print("2. Click góc dưới-phải của vùng cần đo")
    print("3. Tọa độ ROI (trên ảnh gốc) sẽ hiển thị")
    print("4. Nhấn 'r' để reset, 'q' để thoát\n")
    
    cv2.namedWindow("Measure ROI")
    cv2.setMouseCallback("Measure ROI", mouse_callback)
    cv2.imshow("Measure ROI", image_display)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            points.clear()
            # Reset ảnh hiển thị
            if scale_factor != 1.0:
                image_display = cv2.resize(image, (int(orig_w * scale_factor), int(orig_h * scale_factor)))
            else:
                image_display = image.copy()
            cv2.imshow("Measure ROI", image_display)
            print("Đã reset")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
