"""
Script debug để phân tích vùng ROI mã đề và MSSV
"""
import cv2
import numpy as np

# Đọc ảnh vùng mã đề
ma_de_img = cv2.imread('output/06a_ma_de_region.jpg', cv2.IMREAD_GRAYSCALE)
print("=" * 60)
print("PHÂN TÍCH VÙNG MÃ ĐỀ")
print("=" * 60)
print(f"Kích thước: {ma_de_img.shape}")
print(f"Giá trị pixel min: {ma_de_img.min()}, max: {ma_de_img.max()}")
print(f"Giá trị pixel trung bình: {ma_de_img.mean():.2f}")

# Phân ngưỡng để xem
_, binary = cv2.threshold(ma_de_img, 127, 255, cv2.THRESH_BINARY_INV)
white_pixels = cv2.countNonZero(binary)
total_pixels = ma_de_img.shape[0] * ma_de_img.shape[1]
print(f"Pixel trắng (sau threshold): {white_pixels}/{total_pixels} ({white_pixels/total_pixels*100:.1f}%)")

# Tính kích thước bubble
num_digits = 3
choices_per_digit = 10
bubble_height = ma_de_img.shape[0] // choices_per_digit
bubble_width = ma_de_img.shape[1] // num_digits
print(f"\nKích thước bubble: {bubble_width}x{bubble_height}")
print(f"Diện tích bubble: {bubble_width * bubble_height}")
print(f"Ngưỡng 30%: {bubble_width * bubble_height * 0.3:.0f}")

# Phân tích từng chữ số
print("\n" + "=" * 60)
print("PHÂN TÍCH TỪNG CHỮ SỐ")
print("=" * 60)

for digit_idx in range(num_digits):
    print(f"\nChữ số {digit_idx + 1}:")
    pixel_counts = []
    
    for choice_idx in range(choices_per_digit):
        y1 = choice_idx * bubble_height
        y2 = (choice_idx + 1) * bubble_height
        x1 = digit_idx * bubble_width
        x2 = (digit_idx + 1) * bubble_width
        bubble = binary[y1:y2, x1:x2]
        
        count = cv2.countNonZero(bubble)
        pixel_counts.append(count)
    
    # Tìm top 3
    sorted_indices = np.argsort(pixel_counts)[::-1]
    print(f"  Top 3 ô có nhiều pixel nhất:")
    for i in range(3):
        idx = sorted_indices[i]
        print(f"    Ô {idx}: {pixel_counts[idx]} pixels")
    
    max_count = pixel_counts[sorted_indices[0]]
    second_count = pixel_counts[sorted_indices[1]]
    threshold = bubble_width * bubble_height * 0.3
    
    if max_count < threshold:
        print(f"  ⚠️  Không có ô nào đủ pixel (max={max_count}, threshold={threshold:.0f})")
    elif second_count > threshold:
        print(f"  ⚠️  Có 2 ô được tô (max={max_count}, second={second_count}, threshold={threshold:.0f})")
    else:
        print(f"  ✓ Chữ số hợp lệ: {sorted_indices[0]}")

print("\n" + "=" * 60)
print("PHÂN TÍCH VÙNG MÃ SINH VIÊN")
print("=" * 60)

mssv_img = cv2.imread('output/06b_mssv_region.jpg', cv2.IMREAD_GRAYSCALE)
print(f"Kích thước: {mssv_img.shape}")
print(f"Giá trị pixel min: {mssv_img.min()}, max: {mssv_img.max()}")
print(f"Giá trị pixel trung bình: {mssv_img.mean():.2f}")

# Lưu ảnh binary để xem
cv2.imwrite('output/debug_ma_de_binary.jpg', binary)
print("\n✓ Đã lưu ảnh binary: output/debug_ma_de_binary.jpg")
