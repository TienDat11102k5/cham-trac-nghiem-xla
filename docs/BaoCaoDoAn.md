# BÁO CÁO ĐỒ ÁN MÔN XỬ LÝ ẢNH

## HỆ THỐNG CHẤM TRẮC NGHIỆM TỰ ĐỘNG (OMR)

**Optical Mark Recognition System using Python & OpenCV**

---

## 📋 THÔNG TIN ĐỒ ÁN

- **Môn học:** Xử lý Ảnh
- **Đề tài:** Hệ thống chấm trắc nghiệm tự động (OMR)
- **Ngôn ngữ:** Python 3.8+
- **Thư viện chính:** OpenCV, NumPy

### Thành viên nhóm

| STT | Họ và tên | MSSV | Vai trò | Email |
|-----|-----------|------|---------|-------|
| 1   | [Tên TV1] | [MSSV] | Quản trị hệ thống & Kiến trúc | [Email] |
| 2   | [Tên TV2] | [MSSV] | Lập trình viên Cốt lõi 1 | [Email] |
| 3   | [Tên TV3] | [MSSV] | Lập trình viên Cốt lõi 2 | [Email] |
| 4   | [Tên TV4] | [MSSV] | Lập trình viên Cốt lõi 3 | [Email] |
| 5   | [Tên TV5] | [MSSV] | Chuyên viên Dữ liệu & Kiểm thử QA | [Email] |
| 6   | [Tên TV6] | [MSSV] | Chuyên trách Báo cáo khoa học | [Email] |

---

## 1. GIỚI THIỆU

### 1.1. Bối cảnh và động lực

<!-- TODO: Thành viên 6 viết -->
<!-- Nội dung:
- Vấn đề chấm trắc nghiệm thủ công tốn thời gian
- Nhu cầu tự động hóa trong giáo dục
- Ứng dụng thực tế của Computer Vision
-->

### 1.2. Mục tiêu đồ án

<!-- TODO: Thành viên 6 viết -->
<!-- Nội dung:
- Xây dựng hệ thống chấm trắc nghiệm tự động
- Áp dụng các kỹ thuật xử lý ảnh: Canny, Perspective Transform, Thresholding
- Đạt độ chính xác > 95% trên tập test
-->

### 1.3. Phạm vi đồ án

<!-- TODO: Thành viên 6 viết -->
<!-- Nội dung:
- Xử lý ảnh scan/chụp bài thi trắc nghiệm
- Hỗ trợ 40 câu, mỗi câu 4 lựa chọn (A, B, C, D)
- Xuất kết quả dạng JSON/Text
-->

---

## 2. CƠ SỞ LÝ THUYẾT

### 2.1. Xử lý ảnh cơ bản

#### 2.1.1. Chuyển đổi không gian màu (Grayscale)

<!-- TODO: Thành viên 6 viết -->
<!-- Nội dung:
- Công thức chuyển đổi RGB sang Grayscale
- Lý do cần chuyển sang ảnh xám
- Ưu điểm: Giảm độ phức tạp tính toán
-->

#### 2.1.2. Lọc nhiễu (Noise Filtering)

<!-- TODO: Thành viên 6 viết -->
<!-- Nội dung:
- Gaussian Blur: Công thức, kernel, ứng dụng
- Median Filter: Xử lý nhiễu muối tiêu
- Bilateral Filter: Giữ cạnh sắc nét
-->

### 2.2. Phát hiện biên (Edge Detection)

#### 2.2.1. Thuật toán Canny

<!-- TODO: Thành viên 6 viết -->
<!-- Nội dung:
- Các bước của Canny: Gaussian blur, Gradient, Non-maximum suppression, Hysteresis
- Tham số low_threshold và high_threshold
- Ưu điểm: Phát hiện biên chính xác, ít nhiễu
-->

### 2.3. Biến đổi hình học

#### 2.3.1. Phát hiện Contour

<!-- TODO: Thành viên 6 viết -->
<!-- Nội dung:
- cv2.findContours(): Tìm đường viền
- Xấp xỉ đa giác (Polygon Approximation)
- Tìm contour lớn nhất (tờ giấy thi)
-->

#### 2.3.2. Perspective Transform

<!-- TODO: Thành viên 6 viết -->
<!-- Nội dung:
- Ma trận biến đổi phối cảnh
- cv2.getPerspectiveTransform() và cv2.warpPerspective()
- Ứng dụng: Nắn chỉnh ảnh nghiêng về mặt phẳng chuẩn
-->

### 2.4. Phân ngưỡng (Thresholding)

#### 2.4.1. Adaptive Thresholding

<!-- TODO: Thành viên 6 viết -->
<!-- Nội dung:
- Ngưỡng thích nghi theo vùng cục bộ
- Xử lý ảnh có độ sáng không đều
-->

#### 2.4.2. Otsu's Method

<!-- TODO: Thành viên 6 viết -->
<!-- Nội dung:
- Tự động tìm ngưỡng tối ưu
- Dựa trên histogram của ảnh
-->

---

## 3. PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG

### 3.1. Kiến trúc tổng quan

<!-- TODO: Thành viên 1 & 6 viết -->
<!-- Nội dung:
- Sơ đồ kiến trúc 4 module: preprocessing, transform, grader, utils
- Luồng dữ liệu (Data Flow)
- Mô hình MVC/Pipeline
-->

```
[Ảnh đầu vào] 
    ↓
[Preprocessing] → Grayscale, Noise Filtering
    ↓
[Transform] → Edge Detection, Find Corners, Perspective Transform
    ↓
[Grader] → Extract ROI, Segmentation, Scoring
    ↓
[Kết quả: Điểm số + Đáp án]
```

### 3.2. Thiết kế module

#### 3.2.1. Module Preprocessing

<!-- TODO: Thành viên 2 & 6 viết -->
<!-- Nội dung:
- Chức năng: Đọc ảnh, chuyển xám, khử nhiễu
- Input/Output của từng hàm
- Lý do chọn Gaussian filter
-->

#### 3.2.2. Module Transform

<!-- TODO: Thành viên 3 & 6 viết -->
<!-- Nội dung:
- Chức năng: Phát hiện biên, tìm góc, nắn chỉnh
- Thuật toán tìm 4 góc tờ giấy
- Xử lý trường hợp ảnh nghiêng nhiều
-->

#### 3.2.3. Module Grader

<!-- TODO: Thành viên 4 & 6 viết -->
<!-- Nội dung:
- Chức năng: Trích xuất ROI, phân đoạn, chấm điểm
- Thuật toán đếm pixel để xác định ô được tô
- Xử lý trường hợp tô nhiều ô hoặc không tô
-->

### 3.3. Cấu trúc dữ liệu

<!-- TODO: Thành viên 5 & 6 viết -->
<!-- Nội dung:
- Format đáp án chuẩn (JSON)
- Format kết quả đầu ra
- Cấu trúc thư mục dự án
-->

---

## 4. TRIỂN KHAI

### 4.1. Môi trường phát triển

<!-- TODO: Thành viên 1 & 6 viết -->
<!-- Nội dung:
- Python 3.8+
- OpenCV 4.8+, NumPy 1.24+
- Jupyter Notebook cho demo
- Git/GitHub cho quản lý mã nguồn
-->

### 4.2. Chi tiết triển khai

#### 4.2.1. Preprocessing Module

<!-- TODO: Thành viên 2 & 6 viết -->
<!-- Nội dung:
- Code snippet quan trọng
- Giải thích tham số (kernel_size, filter_type)
- Kết quả thực nghiệm
-->

#### 4.2.2. Transform Module

<!-- TODO: Thành viên 3 & 6 viết -->
<!-- Nội dung:
- Code snippet quan trọng
- Giải thích thuật toán tìm góc
- Kết quả thực nghiệm
-->

#### 4.2.3. Grader Module

<!-- TODO: Thành viên 4 & 6 viết -->
<!-- Nội dung:
- Code snippet quan trọng
- Giải thích thuật toán chấm điểm
- Kết quả thực nghiệm
-->

### 4.3. Xử lý ngoại lệ

<!-- TODO: Thành viên 5 & 6 viết -->
<!-- Nội dung:
- Xử lý file không tồn tại
- Xử lý không tìm thấy tờ giấy
- Xử lý ROI không hợp lệ
-->

---

## 5. KIỂM THỬ VÀ ĐÁNH GIÁ

### 5.1. Chiến lược kiểm thử

<!-- TODO: Thành viên 5 & 6 viết -->
<!-- Nội dung:
- Unit Test: Test từng hàm riêng lẻ
- Integration Test: Test toàn bộ pipeline
- Test với mock data và real data
-->

### 5.2. Kết quả kiểm thử

<!-- TODO: Thành viên 5 & 6 viết -->
<!-- Nội dung:
- Bảng kết quả test cases (Pass/Fail)
- Coverage: X% code được test
- Bugs phát hiện và đã fix
-->

### 5.3. Đánh giá độ chính xác

<!-- TODO: Thành viên 5 & 6 viết -->
<!-- Nội dung:
- Tập dữ liệu test: X ảnh
- Accuracy: Y%
- Precision, Recall, F1-Score
- Confusion Matrix
-->

### 5.4. Phân tích lỗi

<!-- TODO: Thành viên 5 & 6 viết -->
<!-- Nội dung:
- Các trường hợp lỗi thường gặp
- Nguyên nhân: Ảnh mờ, nghiêng nhiều, nhiễu cao
- Hướng cải thiện
-->

---

## 6. KẾT QUẢ VÀ THẢO LUẬN

### 6.1. Kết quả đạt được

<!-- TODO: Thành viên 6 viết -->
<!-- Nội dung:
- Hoàn thành hệ thống chấm trắc nghiệm tự động
- Độ chính xác: X%
- Thời gian xử lý: Y giây/ảnh
-->

### 6.2. Ưu điểm

<!-- TODO: Thành viên 6 viết -->
<!-- Nội dung:
- Tự động hóa hoàn toàn
- Độ chính xác cao
- Dễ mở rộng và tùy chỉnh
-->

### 6.3. Hạn chế

<!-- TODO: Thành viên 6 viết -->
<!-- Nội dung:
- Yêu cầu ảnh chất lượng tốt
- Chưa xử lý tốt ảnh nghiêng quá nhiều
- Chưa hỗ trợ nhiều template đề thi
-->

### 6.4. Hướng phát triển

<!-- TODO: Thành viên 6 viết -->
<!-- Nội dung:
- Áp dụng Deep Learning (CNN) để cải thiện độ chính xác
- Hỗ trợ nhiều template đề thi
- Xây dựng Web/Mobile App
- Tích hợp OCR để đọc thông tin sinh viên
-->

---

## 7. KẾT LUẬN

<!-- TODO: Thành viên 6 viết -->
<!-- Nội dung:
- Tóm tắt đồ án
- Kiến thức học được
- Ý nghĩa thực tiễn
-->

---

## 8. TÀI LIỆU THAM KHẢO

<!-- TODO: Thành viên 6 viết -->
<!-- Format chuẩn IEEE hoặc APA -->

1. OpenCV Documentation. (2024). *OpenCV 4.x Documentation*. Retrieved from https://docs.opencv.org/
2. Canny, J. (1986). *A Computational Approach to Edge Detection*. IEEE Transactions on Pattern Analysis and Machine Intelligence.
3. Otsu, N. (1979). *A Threshold Selection Method from Gray-Level Histograms*. IEEE Transactions on Systems, Man, and Cybernetics.
4. [Thêm tài liệu tham khảo khác...]

---

## PHỤ LỤC

### Phụ lục A: Hướng dẫn cài đặt

<!-- TODO: Copy từ README.md -->

### Phụ lục B: Hướng dẫn sử dụng

<!-- TODO: Copy từ README.md -->

### Phụ lục C: Source Code

<!-- TODO: Link đến GitHub repository -->

Repository: https://github.com/TienDat11102k5/cham-trac-nghiem-xla

### Phụ lục D: Kết quả thực nghiệm

<!-- TODO: Thành viên 5 & 6 thêm ảnh kết quả -->
<!-- Bao gồm:
- Ảnh gốc
- Ảnh sau mỗi bước xử lý
- Kết quả chấm điểm
-->

---

**Ngày hoàn thành:** [Ngày/Tháng/Năm]

**Giảng viên hướng dẫn:** [Tên giảng viên]
